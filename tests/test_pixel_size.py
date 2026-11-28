"""The camera's pixel from beads at several stage positions (`smappy.pixel_size`).

A bead field is moved by a known stage pattern under a camera of known pixel, angle and mirror
(`smappy.simulate.stage_scan`), and the numbers must come back -- from the spots, and from the
frames of a written stack through the plugin.
"""
import json

import numpy as np
import pytest

from smappy import microscope_db, pixel_size
from smappy.simulate.stage_scan import raster, simulate_stage_scan, write_tiff


def measured(scan, **options):
    return pixel_size.measure(scan.frame, scan.xy_px, scan.stage_um, scan.shape, **options)


@pytest.mark.parametrize("camera", [
    dict(pixel_nm=(117.0, 113.0), angle_deg=1.5, mirrored=False, backlash_nm=(0.0, 0.0)),
    dict(pixel_nm=(127.0, 116.0), angle_deg=-91.0, mirrored=True, backlash_nm=(300.0, 0.0)),
    dict(pixel_nm=(65.0, 65.0), angle_deg=178.0, mirrored=True, backlash_nm=(-150.0, 0.0)),
])
def test_a_stage_scan_gives_back_the_pixel_the_angle_the_mirror_and_the_backlash(camera):
    step = 0.45 * 128 * np.mean(camera["pixel_nm"]) / 1000.0       # under half a field
    scan = simulate_stage_scan(step_um=(step, step), density_per_um2=12.0 / step ** 2,
                               seed=4, **camera)
    found = measured(scan)
    v = found.values
    assert v["pixel_x_nm"] == pytest.approx(camera["pixel_nm"][0], abs=0.1)
    assert v["pixel_y_nm"] == pytest.approx(camera["pixel_nm"][1], abs=0.1)
    assert v["angle_deg"] == pytest.approx(camera["angle_deg"], abs=0.05)
    assert bool(v["mirrored"]) == camera["mirrored"]
    assert abs(v["skew_deg"]) < 0.1
    # the serpentine reverses x on every row, never y: only x has a backlash to measure
    assert v["backlash_x_nm"] == pytest.approx(camera["backlash_nm"][0], abs=10.0)
    assert np.isnan(v["backlash_y_nm"])
    # the jackknife's errors are of the size the numbers scatter by: a stage noise of 5 nm
    # over 30 frames
    assert 0.0 < v["pixel_x_nm_err"] < 0.1
    assert v["n_frames"] == scan.n_frames
    # every bead is one simulated bead, and no simulated bead two
    linked = found.bead >= 0
    pairs = set(zip(found.bead[linked], scan.bead_true[linked]))
    assert len(pairs) == len({b for b, _ in pairs}) == len({t for _, t in pairs})


def test_without_the_backlash_term_a_serpentine_raster_is_off_by_it():
    """The reason the backlash is fitted: rows approached from both sides sit apart by it."""
    scan = simulate_stage_scan(density_per_um2=0.1, backlash_nm=(400.0, 0.0), seed=2)
    with_term = measured(scan).values
    without = measured(scan, backlash=False).values
    assert with_term["residual_nm"] < 10.0 < without["residual_nm"]


def test_frames_that_share_a_few_beads_are_placed_through_all_of_them():
    """Steps of 80 % of the field: neighbours share three beads or so, and many pairs too few
    to be linked by their vote alone; the groups the vote leaves are joined through every pair
    of frames between them at once, where the frames placed already put them."""
    scan = simulate_stage_scan(step_um=(11.0, 11.0), density_per_um2=0.1, seed=3)
    positions = [scan.xy_px[scan.frame == f] for f in range(scan.n_frames)]
    pairs = pixel_size.frame_pairs(scan.stage_um)
    edges = pixel_size.pair_shifts(positions, pairs, 128.0 * 128.0)
    _, group, _ = pixel_size.solve_offsets(scan.n_frames, edges)
    assert pixel_size._largest(group)[1] <= scan.n_frames / 2    # the vote alone: half
    found = measured(scan)
    assert found.values["n_frames"] == scan.n_frames
    assert found.values["pixel_x_nm"] == pytest.approx(117.0, abs=0.1)
    assert found.values["pixel_y_nm"] == pytest.approx(113.0, abs=0.1)


def test_a_wrong_expected_pixel_costs_nothing_and_a_right_one_the_same():
    scan = simulate_stage_scan(density_per_um2=0.1, seed=5)
    plain = measured(scan).values["pixel_x_nm"]
    for hint in (115.0, 70.0, 250.0):
        assert measured(scan, pixel_hint_nm=hint).values["pixel_x_nm"] == pytest.approx(plain)


def test_steps_larger_than_the_beads_can_bridge_are_refused_in_words():
    scan = simulate_stage_scan(step_um=(12.0, 12.0), density_per_um2=0.05)
    with pytest.raises(ValueError, match="too large for the field"):
        measured(scan)


def test_a_stage_that_moved_along_one_line_is_refused():
    scan = simulate_stage_scan(n_x=12, n_y=1, density_per_um2=0.1)
    with pytest.raises(ValueError, match="one line"):
        measured(scan)


def test_where_the_approach_is_the_row_backlash_cannot_be_told_from_the_row_step():
    """x approached forward on every frame of the first row and back on every frame of the
    second (the second row half a step over, so its first frame too): the direction is the row,
    and a backlash term in x would take the step in y with it.  It is left out."""
    stage = raster(6, 2, (8.0, 8.0))
    stage[6:, 0] -= 4.0
    directions = pixel_size.approach_directions(stage)
    np.testing.assert_array_equal(directions[:, 0], [1] * 6 + [-1] * 6)
    rows = np.arange(len(stage))
    assert pixel_size.backlash_axes(directions, rows) == [0]
    assert pixel_size.backlash_axes(directions, rows, stage) == []
    found = measured(simulate_stage_scan(density_per_um2=0.1, stage_um=stage))
    assert found.values["pixel_y_nm"] == pytest.approx(113.0, abs=0.3)
    assert np.isnan(found.values["backlash_x_nm"])
    # a serpentine's second row starts where the first ended, approached forward: that one
    # frame tells the backlash from the step
    plain = raster(6, 2, (8.0, 8.0))
    assert pixel_size.backlash_axes(pixel_size.approach_directions(plain), rows, plain) == [0]


def test_the_approach_direction_is_the_last_move_along_each_axis():
    stage = np.array([[0, 0], [1, 0], [2, 0], [2, 1], [1, 1], [1, 1], [0, 2]], float)
    np.testing.assert_array_equal(pixel_size.approach_directions(stage)[:, 0],
                                  [1, 1, 1, 1, -1, -1, -1])
    np.testing.assert_array_equal(pixel_size.approach_directions(stage)[:, 1],
                                  [1, 1, 1, 1, 1, 1, 1])


def test_decompose_reads_back_the_camera_it_was_made_from():
    for pixel, angle, mirrored in (((100.0, 100.0), 0.0, False), ((127.0, 116.0), 30.0, True),
                                   ((65.0, 66.0), -170.0, False)):
        v = pixel_size.decompose(pixel_size.camera_matrix(pixel, angle, mirrored))
        assert (v["pixel_x_nm"], v["pixel_y_nm"]) == pytest.approx(pixel)
        assert v["angle_deg"] == pytest.approx(angle)
        assert v["mirrored"] == float(mirrored) and v["skew_deg"] == pytest.approx(0.0, abs=1e-9)


def test_a_vote_finds_the_shared_beads_among_many_that_are_not():
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 200, (60, 2))
    shift = np.array([37.3, -81.6])
    b = np.vstack([a[:5] + shift + rng.normal(0, 0.05, (5, 2)), rng.uniform(0, 200, (55, 2))])
    found, votes = pixel_size.frame_shift(a, b)
    assert votes == 5 and found == pytest.approx(shift, abs=0.1)
    assert votes >= pixel_size.chance_votes(60, 60, 200.0 ** 2, pixel_size.TOLERANCE_PX,
                                            pixel_size.FALSE_LINK / 100)


def test_the_stage_tags_are_found_by_micro_managers_name_or_as_a_pair():
    frames = np.arange(3)
    tags = {"frame": frames, "XPositionUm": frames * 10.0, "YPositionUm": frames * 0.0,
            "Camera-Temp": frames * 1.0}
    np.testing.assert_allclose(pixel_size.stage_positions(tags, 3)[:, 0], [0, 10, 20])
    other = {"frame": frames, "Stage-X": frames * 1.0, "Stage-Y": frames * 2.0}
    assert pixel_size.stage_columns(other) == ("Stage-X", "Stage-Y")
    with pytest.raises(ValueError, match="Camera-Temp"):
        pixel_size.stage_columns({"frame": frames, "Camera-Temp": frames * 1.0})
    nm = pixel_size.stage_positions({"frame": frames, "Stage-X": frames * 1000.0,
                                     "Stage-Y": frames * 0.0}, 3, unit_um=1e-3)
    np.testing.assert_allclose(nm[:, 0], [0, 1, 2])


# ------------------------------------------------------------------ the microscope's entry

LENS = {"3DA": {"tag": "Lens-Position", "states": {"on": "1", "off": "0"}}}


def scope(**parameters):
    raw = {"name": "scope", "identify": {"Scope-Serial": "test-1"}, "devices": LENS,
           "parameters": dict({"modality": {"device": "3DA",
                                            "values": {"on": "astigmatic", "off": "2D"}}},
                              **parameters)}
    return microscope_db.Microscope.from_dict(raw)


def test_a_measured_pixel_replaces_the_value_of_the_state_it_was_measured_in_only():
    entry = scope(pixel_nm={"device": "3DA", "values": {"on": [127, 116], "off": [127, 127]}})
    new, words = microscope_db.with_pixel(entry, {"3DA": "on"}, (126.94, 116.36))
    assert new.parameters["pixel_nm"]["values"] == {"on": [126.9, 116.4], "off": [127, 127]}
    assert "3DA on" in words and entry.parameters["pixel_nm"]["values"]["on"] == [127, 116]


def test_a_fixed_pixel_becomes_one_per_state_of_the_3d_lens():
    new, _ = microscope_db.with_pixel(scope(pixel_nm=127), {"3DA": "on"}, (126.0, 116.0))
    assert new.parameters["pixel_nm"] == {"device": "3DA",
                                          "values": {"on": [126.0, 116.0], "off": 127}}
    bare = microscope_db.Microscope(name="plain")
    new, _ = microscope_db.with_pixel(bare, {}, (100.04, 100.0))
    assert new.parameters["pixel_nm"] == 100.0


@pytest.fixture
def config(tmp_path, monkeypatch):
    from smappy import camera_db
    from smappy import config as settings
    monkeypatch.setenv("SMAPPY_CONFIG_DIR", str(tmp_path / "config"))
    settings.load(reload=True)
    camera_db.database(reload=True)
    microscope_db.database(reload=True)
    yield tmp_path / "config"
    monkeypatch.undo()
    settings.load(reload=True)
    camera_db.database(reload=True)
    microscope_db.database(reload=True)


def test_the_plugin_measures_a_written_stack_and_enters_the_pixel_for_its_state(config, tmp_path):
    from smappy.plugins import Context
    from smappy.plugins.pixel_size import (PixelSizeCalibration, PixelSizeSettings, last_entry)
    microscope_db.save_microscope(scope(pixel_nm={"device": "3DA", "values": {
        "on": [127, 116], "off": [127, 127]}}))
    scan = simulate_stage_scan(images=True, density_per_um2=0.1, pixel_nm=(126.0, 115.0),
                               backlash_nm=(250.0, 0.0), seed=6)
    path = write_tiff(scan, tmp_path / "beads.tif",
                      tags={"Scope-Serial": "test-1", "Lens-Position": "1"})
    plugin = PixelSizeCalibration()
    settings = PixelSizeSettings(path=str(path))
    notice = plugin.describe(None, settings).notice
    assert notice.window == "microscope manager" and "127 x 116" in notice.text
    result = plugin.run(Context(), settings)
    assert result.data["pixel_x_nm"] == pytest.approx(126.0, abs=0.15)
    assert result.data["pixel_y_nm"] == pytest.approx(115.0, abs=0.15)
    assert result.data["backlash_x_nm"] == pytest.approx(250.0, abs=15.0)
    assert "-0.8 %" in result.text                   # against the database's 127
    notice = plugin.describe(None, settings).notice
    assert notice.window == "pixel size entry" and "scope with 3DA on" in notice.text
    entry, words = last_entry(settings)
    assert entry.parameters["pixel_nm"]["values"]["on"] == pytest.approx([126.0, 115.0], abs=0.2)
    assert microscope_db.database().get("scope").parameters["pixel_nm"]["values"]["on"] == \
        [127, 116]                                   # shown, not saved
    plugin.run(Context(), PixelSizeSettings(path=str(path), enter=True))
    kept = microscope_db.database().get("scope").parameters["pixel_nm"]["values"]
    assert kept["on"] == pytest.approx([126.0, 115.0], abs=0.2) and kept["off"] == [127, 127]


def test_the_notices_window_shows_the_measured_pixel_in_the_manager_unsaved(config, tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from smappy.gui.app import ControlWindow, RenderWindow
    from smappy.plugins import Context
    from smappy.plugins.pixel_size import PixelSizeCalibration, PixelSizeSettings
    from smappy.session import Session
    QApplication.instance() or QApplication([])
    microscope_db.save_microscope(scope(pixel_nm=127))
    scan = simulate_stage_scan(images=True, density_per_um2=0.1, pixel_nm=(126.0, 115.0),
                               seed=7)
    path = write_tiff(scan, tmp_path / "beads.tif",
                      tags={"Scope-Serial": "test-1", "Lens-Position": "0"})
    session = Session()
    window = ControlWindow(session, RenderWindow(session))
    try:
        session.window_openers["pixel size entry"](PixelSizeSettings(path=str(path)))
        assert "run the calibration first" in window._microscope_manager.message.text()
        PixelSizeCalibration().run(Context(), PixelSizeSettings(path=str(path)))
        session.window_openers["pixel size entry"](PixelSizeSettings(path=str(path)))
        manager = window._microscope_manager
        assert manager.isVisible() and "3DA off" in manager.message.text()
        shown = manager.entry().parameters["pixel_nm"]
        assert shown["device"] == "3DA" and shown["values"]["on"] == 127
        assert shown["values"]["off"] == pytest.approx([126.0, 115.0], abs=0.2)
        assert microscope_db.database().get("scope").parameters["pixel_nm"] == 127
    finally:
        window.close()


def test_the_stack_written_for_it_carries_the_stage_in_each_frame(tmp_path):
    import tifffile
    scan = simulate_stage_scan(n_x=2, n_y=2, images=True, density_per_um2=0.1)
    with tifffile.TiffFile(write_tiff(scan, tmp_path / "s.tif")) as tf:
        md = tf.pages[3].tags["MicroManagerMetadata"].value
    md = md if isinstance(md, dict) else json.loads(md)
    assert (md["XPositionUm"], md["YPositionUm"]) == tuple(scan.stage_um[3])
