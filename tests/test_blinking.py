"""Blinking, and the frame time that turns its frames into seconds.

The counts are checked exactly on frames written by hand, and the means on
the kinetics of the simulator (`simulate.kinetics`) with known on- and
off-times and bleaching: an exponential on-time of ``tau`` frames, starting
anywhere in a frame, touches ``tau + 1`` frames on average.  A dark time
that leaves no frame wholly dark is not seen -- the two blinks are one run
-- and past the first dark frame an exponential dark time starts afresh, so
the dark frames seen are ``1 + floor(r)`` with ``r`` exponential of mean
``tau``: ``1 + 1 / (exp(1/tau) - 1)``, about ``tau + 0.5``, on average.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from smappy import camera_db
from smappy.camera_db import Camera
from smappy.io.formats import FileInfo, _sml_camera
from smappy.locs import Localizations
from smappy.metadata import CameraMetadata, file_frame_time, frame_time_ms
from smappy.plugins import Context
from smappy.plugins.blinking import (Blinking, blinks_of, bursts,
                                     dark_frames, molecule_blinking, region_blinking)
from smappy.simulate.kinetics import blink, emission
from smappy.simulate.settings import BlinkingSettings as Kinetics

ON, OFF, BLEACH = 2.0, 30.0, 0.25


# ------------------------------------------------------------- by hand

def test_blinks_are_runs_of_consecutive_frames_and_dark_times_lie_between():
    found = blinks_of([50, 0, 1, 2, 5, 6])
    assert found.first.tolist() == [0, 5, 50] and found.last.tolist() == [2, 6, 50]
    assert found.on_frames.tolist() == [3, 2, 1]
    gaps = dark_frames(found)
    assert np.isnan(gaps[0]) and gaps[1:].tolist() == [2, 43]


def test_a_merged_blink_bridges_the_frame_the_fit_missed():
    found = blinks_of([0, 2, 3, 10], blink=[1, 1, 1, 2])
    assert found.on_frames.tolist() == [4, 1]
    assert dark_frames(found)[1] == 6


def test_blinks_on_together_have_no_dark_time_between_them():
    # two fluorophores of one region: 0-9 and 3-5, then one at 20
    found = blinks_of([0, 9, 3, 4, 5, 20], blink=[1, 1, 2, 2, 2, 3])
    gaps = dark_frames(found)
    assert np.isnan(gaps[1]) and gaps[2] == 10      # after frame 9, not frame 5


def test_bursts_split_at_a_dark_time_longer_than_t():
    found = blinks_of([0, 1, 2, 5, 6, 50])
    assert bursts(found, 10).tolist() == [0, 0, 1]
    assert bursts(found, 43).tolist() == [0, 0, 0]
    assert bursts(found, 42).tolist() == [0, 0, 1]
    values = region_blinking([0, 1, 2, 5, 6, 50], dark_time=10, frame_time_ms=20.0)
    assert values["n_localizations"] == 6 and values["n_blinks"] == 3
    assert values["n_bursts"] == 2
    assert values["mean_on_frames"] == 2.0 and values["mean_dark_frames"] == 2.0
    assert values["mean_on_s"] == pytest.approx(0.04)
    assert np.isnan(region_blinking([0, 1], frame_time_ms=None)["mean_on_s"])


def test_molecules_keep_their_blinks_apart():
    found = blinks_of([0, 1, 0, 1], molecule=[1, 1, 2, 2])
    assert len(found) == 2 and np.isnan(dark_frames(found)).all()


def test_an_empty_region_is_an_answer():
    values = region_blinking(np.zeros(0))
    assert values["n_blinks"] == 0 and values["n_bursts"] == 0
    assert np.isnan(values["mean_on_frames"])


# ------------------------------------------------------- simulated kinetics

def kinetics(n=3000, seed=1, bleaching=BLEACH, n_frames=1_000_000):
    """Frames, fluorophore and blink of every localization: decaying
    activation, so the off-times are the exponential ones asked for."""
    rng = np.random.default_rng(seed)
    settings = Kinetics(on_time=ON, off_time=OFF, bleaching=bleaching,
                        activation="decay", photons=1000, photons_std=0)
    blinks = blink(n, n_frames, settings, rng)
    out = emission(blinks, n_frames)
    order = np.lexsort((out.frame, out.owner))
    return out.frame[order], out.owner[order], blinks


def test_the_on_and_dark_times_of_simulated_kinetics_are_the_ones_simulated():
    frame, owner, _ = kinetics()
    # spread out so each fluorophore is a molecule of its own
    x = (owner % 100) * 1000.0
    y = (owner // 100) * 1000.0
    found = molecule_blinking(frame, x, y, dark_time=1000, distance_nm=50)
    assert np.mean(found.on_frames) == pytest.approx(ON + 1, rel=0.05)
    seen = 1 + 1 / np.expm1(1 / OFF)
    assert np.mean(found.dark_frames) == pytest.approx(seen, rel=0.03)
    # geometric with p = 0.25; a dark time under a frame merges two blinks
    assert np.mean(found.blinks_per_molecule) == pytest.approx(1 / BLEACH, rel=0.05)
    assert len(found.blinks_per_molecule) == len(np.unique(owner))


def test_bursts_count_the_molecules_of_a_region_that_came_on_one_after_another():
    frame, owner, _ = kinetics(n=12, seed=3)
    # each fluorophore 10^7 frames after the one before
    frame = frame + owner * 10_000_000
    values = region_blinking(frame, dark_time=1000)
    assert values["n_bursts"] == len(np.unique(owner))
    assert values["n_blinks"] >= values["n_bursts"]


def test_the_plugin_reads_the_files_frame_time_and_measures_per_region():
    frame, owner, _ = kinetics(n=200)
    locs = Localizations({"frame": frame, "x_nm": (owner % 20) * 1000.0,
                          "y_nm": (owner // 20) * 1000.0},
                         {"camera": {"exposure_ms": 30.0, "frame_time_ms": 50.0}})
    result = Blinking()(locs, time_axis="seconds")
    assert result.values["mean_on_s"] == pytest.approx(
        result.values["mean_on_frames"] * 0.05)
    (found,) = result.data["files"]
    assert found.frame_time_ms == 50.0
    assert np.mean(found.on_frames) == pytest.approx(ON + 1, rel=0.1)
    region = Context(locs=locs, region_id="1")
    assert Blinking()(ctx=region).plot is None


def a_session(n=100, grouped=True):
    from test_roi_session import rois_of          # noqa: E402
    frame, owner, _ = kinetics(n=n, seed=5)
    rng = np.random.default_rng(0)
    m = len(frame)
    locs = Localizations({
        "frame": frame, "x_nm": (owner % 10) * 1000.0 + rng.normal(0, 3, m),
        "y_nm": (owner // 10) * 1000.0 + rng.normal(0, 3, m),
        "photons": np.full(m, 1000.0, np.float32),
        "xy_err_nm": np.full(m, 5.0, np.float32)},
        {"units": "nm", "camera": {"exposure_ms": 10.0}})
    project, source = rois_of(locs, grouped=grouped)
    return project, source, frame, owner


def test_a_layer_that_merges_blinks_is_measured_by_its_blinks():
    project, _, frame, owner = a_session()
    session = project.session
    settings = Blinking.Settings(dark_time=1000)        # the off-time is 30
    merged = Blinking().run(session.context(0), settings)
    session.show_grouped(0, False)
    runs = Blinking().run(session.context(0), settings)
    assert merged.values["n_localizations"] == runs.values["n_localizations"] == len(frame)
    # merging bridges one dark frame, so it finds fewer and longer blinks
    assert merged.values["n_blinks"] < runs.values["n_blinks"]
    assert merged.values["n_bursts"] == runs.values["n_bursts"] == len(np.unique(owner))
    assert merged.values["mean_on_frames"] > runs.values["mean_on_frames"]
    assert merged.values["mean_on_s"] == pytest.approx(
        merged.values["mean_on_frames"] * 0.01)


def test_in_an_roi_it_counts_the_blinks_and_bursts_of_what_is_inside():
    project, source, frame, owner = a_session()
    roi = project.add_roi(source.id, [3000.0, 2000.0])         # fluorophore 23
    result = Blinking().run(project.region_context(roi), Blinking.Settings())
    mine = frame[owner == 23]
    assert result.values["n_localizations"] == len(mine)
    assert result.values["n_bursts"] == 1
    # the blinks as the layer merged them
    ids = np.asarray(project.session.locs["group_id"])[owner == 23]
    assert result.values["n_blinks"] == len(np.unique(ids))


# --------------------------------------------------------------- frame time

def test_the_frame_time_falls_back_to_the_exposure():
    assert frame_time_ms({"exposure_ms": 30.0}) == 30.0
    assert frame_time_ms({"exposure_ms": 30.0, "frame_time_ms": 31.5}) == 31.5
    assert frame_time_ms({"exposure_ms": 0.0}) is None and frame_time_ms(None) is None
    assert CameraMetadata(exposure_ms=20.0).frame_time == 20.0


def source(**tags):
    return SimpleNamespace(summary=tags, mm_metadata={}, camera=None)


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("SMAPPY_CONFIG_DIR", str(tmp_path))
    camera_db.database(reload=True)
    yield tmp_path
    monkeypatch.undo()
    camera_db.database(reload=True)


def test_the_frame_time_is_read_from_the_files_metadata(config):
    from smappy.io.tiff import camera_metadata
    cam = camera_metadata(source(**{"Core-Camera": "Cam", "Exposure-ms": "30",
                                    "Cam-ActualInterval-ms": "31.5"}), require=False)
    assert cam.exposure_ms == 30.0 and cam.frame_time_ms == 31.5
    # a time-lapse interval longer than the exposure is the frame time ...
    cam = camera_metadata(source(**{"Exposure-ms": "30", "Interval_ms": "100"}),
                          require=False)
    assert cam.frame_time == 100.0
    # ... and one of 0 says the camera ran as fast as it could
    cam = camera_metadata(source(**{"Exposure-ms": "30", "Interval_ms": "0"}),
                          require=False)
    assert cam.frame_time_ms is None and cam.frame_time == 30.0


def test_a_frame_time_typed_into_the_fit_is_kept_with_the_table(config, tmp_path):
    from smappy.io.hdf5 import load_localizations, save_localizations
    from smappy.plugins.fit import CameraSettings
    cam = CameraSettings(frame_time_ms=40.0).resolve(
        source(**{"Exposure-ms": "30"}), require=False)
    assert cam.frame_time_ms == 40.0
    locs = Localizations({"frame": np.arange(3), "x_nm": np.zeros(3),
                          "y_nm": np.zeros(3)}, {"camera": cam.to_dict()})
    path = save_localizations(tmp_path / "t.h5", locs)
    assert file_frame_time(load_localizations(path).metadata) == 40.0


def test_each_file_of_a_session_keeps_its_own_frame_time():
    from smappy.session import Session

    def table(ms):
        return Localizations({"frame": np.arange(3), "x_nm": np.zeros(3),
                              "y_nm": np.zeros(3)}, {"camera": {"exposure_ms": ms}})
    session = Session()
    session.add_file(table(20.0), FileInfo("a", "a", "test"))
    session.add_file(table(50.0), FileInfo("b", "b", "test"), append=True)
    assert file_frame_time(session.locs.metadata, 0) == 20.0
    assert file_frame_time(session.locs.metadata, 1) == 50.0


@pytest.fixture
def app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_the_editor_sets_a_cameras_frame_time(app, config):
    from smappy.gui.camera_editor import CameraEditor
    tags = {"Core-Camera": "Prime95B", "Prime95B-SerialNumber": "XY123",
            "Prime95B-Exposure": "20", "PixelSizeUm": "0.11"}
    editor = CameraEditor(tags)
    editor.rows["conversion"].value.setText("1.2")
    editor.rows["offset"].value.setText("100")
    row = editor.rows["frame_time_ms"]
    row.source.setCurrentIndex(row.source.findData("fixed"))
    row.value.setText("25")
    editor.changed()
    editor.save()
    found = camera_db.database().resolve(tags).camera_metadata()
    assert found.frame_time_ms == 25.0 and found.exposure_ms == 20.0


def test_a_database_entry_written_with_the_old_name_is_read():
    camera = Camera.from_dict({"name": "old", "parameters": {
        "frame_interval_ms": {"fixed": 12.0}}})
    assert "frame_time_ms" in camera.parameters


def test_a_smap_files_frame_time_is_its_timediff_unless_a_placeholder():
    assert _sml_camera({"exposure": 30.0, "timediff": 33.0}) == {
        "exposure_ms": 30.0, "frame_time_ms": 33.0}
    assert _sml_camera({"exposure": 30.0, "timediff": 1.0}) == {"exposure_ms": 30.0}
    assert _sml_camera({"pixelsize_um": 0.117})["pixelsize_um"] == 0.117
