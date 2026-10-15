"""Transform: typed operations, files, images and another file's drift.

The typed operations are checked against coordinates worked out by hand, the
rest against what they are for: a _2ct.h5 brings the second camera onto the
first, an image moved by placement lands where the localizations moved by the
same transformation land, another file's drift corrects a file of the same
frames and refuses one of other frames.
"""
import dataclasses

import numpy as np
import pytest

from smappy import plugins
from smappy.drift import Drift
from smappy.images import ImageData
from smappy.locs import Localizations
from smappy.plugins import Context
from smappy.plugins.transform import TransformSettings, TypedSettings, settings_for
from smappy.render import FieldOfView
from smappy.session import Session
from smappy.transformation import (Transformation, check_same_frames,
                                   load_transformation, save_transformation,
                                   transform_table, typed)

PATH = "Analysis/Process/Transform"


def points():
    """Four points and a z, easy to follow by hand."""
    return Localizations({"x_nm": np.array([0.0, 100.0, 0.0, 100.0]),
                          "y_nm": np.array([0.0, 0.0, 50.0, 50.0]),
                          "z_nm": np.array([10.0, 20.0, 30.0, 40.0]),
                          "frame": np.arange(4)}, {})


def run_typed(locs, **values):
    s = TransformSettings(typed=TypedSettings(**values))
    return plugins.get(PATH)().run(Context(locs=locs), s).locs


def xyz(locs):
    return np.column_stack([np.asarray(locs[c], float) for c in ("x_nm", "y_nm", "z_nm")])


# ------------------------------------------------------------------- typed
def test_a_shift_moves_x_y_and_z():
    out = run_typed(points(), shift_x_nm=5, shift_y_nm=-7, shift_z_nm=100)
    np.testing.assert_allclose(xyz(out), xyz(points()) + [5, -7, 100])


def test_a_rotation_by_90_degrees_about_the_origin_is_smaps():
    # SMAP: x' = cos x + sin y, y' = -sin x + cos y -- (x, y) -> (y, -x)
    out = run_typed(points(), angle_deg=90, centre="origin")
    np.testing.assert_allclose(xyz(out), [[0, 0, 10], [0, -100, 20],
                                          [50, 0, 30], [50, -100, 40]], atol=1e-9)


def test_a_rotation_keeps_the_middle_of_the_target():
    # middle of the box: (50, 25); (0, 0) is (-50, -25) from it -> (-25, 50)
    out = run_typed(points(), angle_deg=90)
    np.testing.assert_allclose(xyz(out)[:, :2], [[25, 75], [25, -25],
                                                 [75, 75], [75, -25]], atol=1e-9)


def test_a_scale_about_a_given_centre():
    out = run_typed(points(), scale=2, scale_y=3, centre="given",
                    centre_x_nm=100, centre_y_nm=0)
    np.testing.assert_allclose(xyz(out)[:, :2], [[-100, 0], [100, 0],
                                                 [-100, 150], [100, 150]])
    np.testing.assert_allclose(xyz(out)[:, 2], xyz(points())[:, 2])


@pytest.mark.parametrize("mirror, expected", [
    ("x", [[100, 0], [0, 0], [100, 50], [0, 50]]),
    ("y", [[0, 50], [100, 50], [0, 0], [100, 0]]),
    ("both", [[100, 50], [0, 50], [100, 0], [0, 0]]),
])
def test_a_mirror_turns_the_target_over_its_middle(mirror, expected):
    out = run_typed(points(), mirror=mirror)
    np.testing.assert_allclose(xyz(out)[:, :2], expected)


def test_the_typed_operations_compose_mirror_scale_rotate_then_shift():
    # about c = (0, 0): mirror x -> (-x, y); scale 2 -> (-2x, 2y);
    # rotate 90 -> (2y, 2x); shift (+1, +2)
    out = run_typed(points(), mirror="x", scale=2, angle_deg=90, centre="origin",
                    shift_x_nm=1, shift_y_nm=2)
    x, y = xyz(points())[:, 0], xyz(points())[:, 1]
    np.testing.assert_allclose(xyz(out)[:, 0], 2 * y + 1, atol=1e-9)
    np.testing.assert_allclose(xyz(out)[:, 1], 2 * x + 2, atol=1e-9)


def test_the_inverse_takes_the_points_back():
    t = typed((3, 4, 5), 33.0, 1.5, None, "y", (10, 20))
    x, y, z = t.apply([1.0, 2.0], [3.0, 4.0], [5.0, 6.0])
    back = t.inverse().apply(x, y, z)
    np.testing.assert_allclose(np.column_stack(back), [[1, 3, 5], [2, 4, 6]])


def test_one_channel_of_one_file_moves_and_the_rest_stays():
    locs = Localizations({"x_nm": np.zeros(6), "y_nm": np.zeros(6),
                          "frame": np.arange(6),
                          "filenumber": np.array([0, 0, 0, 1, 1, 1]),
                          "channel": np.array([1, 2, 1, 2, 1, 2])}, {})
    s = TransformSettings(file=1, channel=2, typed=TypedSettings(shift_x_nm=10))
    out = plugins.get(PATH)().run(Context(locs=locs), s).locs
    np.testing.assert_allclose(out["x_nm"], [0, 0, 0, 10, 0, 10])
    record = out.metadata["transformations"][-1]
    assert record["n"] == 2 and record["transformation"]["kind"] == "affine"


# --------------------------------------------------------- _2ct.h5 and files
def _two_cameras(h_px, pixelsize=100.0):
    """A simulated reference channel and the same molecules as the second
    camera sees them: ``h_px`` maps second -> first, in chip pixels."""
    from smappy.simulate import simulate
    from smappy.calibrate.dual import map_points
    first = simulate()
    n = len(first)
    xy = np.column_stack([first["x_nm"], first["y_nm"]]) / pixelsize
    second = map_points(np.linalg.inv(h_px), xy) * pixelsize
    columns = {"x_nm": np.r_[first["x_nm"], second[:, 0]],
               "y_nm": np.r_[first["y_nm"], second[:, 1]],
               "frame": np.r_[first["frame"], first["frame"]],
               "channel": np.r_[np.ones(n, int), np.full(n, 2)]}
    return Localizations(columns, {"pixelsize_nm": pixelsize}), n


@pytest.mark.parametrize("perspective", [0.0, 2e-5])
def test_a_2ct_file_applied_to_the_second_camera_matches_the_first(tmp_path, perspective):
    from smappy.calibrate.dual import ChannelTransform
    from smappy.calibrate.transform import save_channel_transform
    from smappy.simulate.camera import dual_transformation
    h = dual_transformation(100)
    h[2, 0] = perspective                  # 0: affine; else a projective map
    path = tmp_path / "cams_2ct.h5"
    save_channel_transform(path, ChannelTransform(h, {}, {}))
    locs, n = _two_cameras(h)
    s = TransformSettings(file=-1, channel=2, source="file", path=str(path))
    result = plugins.get(PATH)().run(Context(locs=locs), s)
    out = result.locs
    assert result.data["transformation"].kind == ("affine" if not perspective
                                                  else "projective")
    for column in ("x_nm", "y_nm"):
        np.testing.assert_allclose(np.asarray(out[column])[n:],
                                   np.asarray(out[column])[:n], atol=1e-3)


def test_a_transformation_round_trips_through_its_file(tmp_path):
    t = typed((12.5, -3, 40), 17.0, 1.02, 0.98, "x", (1000, 2000))
    path = save_transformation(tmp_path / "a_T.h5", t, name="cam2")
    back = load_transformation(path)
    np.testing.assert_allclose(back.matrix, t.matrix)
    assert back.kind == "affine" and back.origin["typed"]["angle_deg"] == 17.0
    with pytest.raises(FileExistsError):
        save_transformation(path, t)
    # a projective one too, and through the plugin's *write to*
    p = Transformation("projective", np.array([[1, 0.01, 5], [0, 1, 7], [1e-6, 0, 1]]))
    path = save_transformation(tmp_path / "p_T.h5", p)
    np.testing.assert_allclose(load_transformation(path).matrix, p.matrix)


def test_the_plugin_writes_and_keeps_what_it_applied(tmp_path):
    session = Session(points())
    s = TransformSettings(typed=TypedSettings(shift_x_nm=10), keep_as="ten",
                          write_to=str(tmp_path / "ten_T.h5"))
    session.run(plugins.get(PATH)(), s)
    assert "ten" in session.transformations
    np.testing.assert_allclose(load_transformation(tmp_path / "ten_T.h5").matrix,
                               session.transformations["ten"].matrix)
    # applied again by name, backwards: back where it started
    session.run(plugins.get(PATH)(), dataclasses.replace(
        settings_for("ten"), invert=True))
    np.testing.assert_allclose(session.locs["x_nm"], points()["x_nm"])
    assert session.locs.metadata["transformations"][-1]["name"] == "ten"


def test_named_transformations_are_saved_with_the_file_and_not_the_workspace(tmp_path):
    session = Session(points())
    session.add_transformation("cam2", typed((5, 0, 0)))
    path = session.save(tmp_path / "kept.hdf5")
    other = Session()
    other.load(path)
    np.testing.assert_allclose(other.transformations["cam2"].matrix,
                               session.transformations["cam2"].matrix)


# ------------------------------------------------------------------ images
def test_an_image_moved_by_placement_lands_where_the_localizations_land():
    """Bright pixels at the localizations, moved by one transformation as an
    image and as localizations: the moved image is bright where the moved
    localizations are."""
    rng = np.random.default_rng(1)
    pixel = 20.0
    plane = np.zeros((60, 80), np.float32)
    ij = np.column_stack([rng.integers(5, 55, 30), rng.integers(5, 75, 30)])
    plane[ij[:, 0], ij[:, 1]] = 1.0
    x = (ij[:, 1] + 0.5) * pixel + 300.0      # the pixel centres, image at x0=300
    y = (ij[:, 0] + 0.5) * pixel + 100.0
    locs = Localizations({"x_nm": x, "y_nm": y, "frame": np.zeros(len(x), int)}, {})
    session = Session(locs)
    session.add_image(ImageData(plane, pixel, x0=300.0, y0=100.0, name="wf"))
    s = TransformSettings(typed=TypedSettings(angle_deg=30, scale=1.25,
                                              mirror="x", shift_x_nm=-70,
                                              centre="given", centre_x_nm=900,
                                              centre_y_nm=700))
    plugin = plugins.get(PATH)()
    session.run(plugin, dataclasses.replace(s, target=2))    # the image layer
    session.run(plugin, s)                                   # the localizations
    image = session.layers[1].image
    assert image.placement is not None
    fov = FieldOfView(-1500.0, -1500.0, 4.0, 1250, 1250)
    picture = image.resample(fov).weight
    col = np.floor((np.asarray(session.locs["x_nm"]) - fov.x0) / fov.pixelsize).astype(int)
    row = np.floor((np.asarray(session.locs["y_nm"]) - fov.y0) / fov.pixelsize).astype(int)
    assert picture[row, col].min() == 1.0
    assert picture.sum() < 0.1 * picture.size      # and not everywhere


def test_an_image_move_is_one_undo_step_and_a_projective_map_is_refused(tmp_path):
    session = Session(points())
    image = ImageData(np.ones((10, 10), np.float32), 10.0, name="wf")
    session.add_image(image)
    plugin = plugins.get(PATH)()
    session.run(plugin, TransformSettings(target=2, typed=TypedSettings(shift_x_nm=40)))
    assert image.bounds[0] == pytest.approx(40.0)
    session.undo()
    assert image.placement is None and image.bounds[0] == 0.0
    session.redo()
    assert image.bounds[0] == pytest.approx(40.0)
    p = Transformation("projective", np.array([[1, 0, 0], [0, 1, 0], [1e-4, 0, 1]]))
    session.add_transformation("bent", p)
    with pytest.raises(ValueError, match="resampled"):
        session.run(plugin, settings_for("bent", target=2))
    with pytest.raises(ValueError, match="no z"):
        session.run(plugin, TransformSettings(target=2, typed=TypedSettings(shift_z_nm=5)))


def test_an_image_placement_is_saved_with_the_file(tmp_path):
    session = Session(points())
    image = ImageData(np.arange(12, dtype=np.float32).reshape(3, 4), 10.0, name="wf")
    session.add_image(image)
    session.run(plugins.get(PATH)(), TransformSettings(
        target=2, typed=TypedSettings(angle_deg=90, centre="origin")))
    path = session.save(tmp_path / "moved.hdf5")
    from smappy.io.hdf5 import load_images
    back = load_images(path)[0]
    np.testing.assert_allclose(back.placement, image.placement)


# ------------------------------------------------------------------- drift
def _drift_file(tmp_path, n_frames, name="cam1_driftc.hdf5"):
    from smappy.drift import save_drift_corrected
    frames = np.arange(n_frames)
    curve = np.column_stack([0.5 * frames, -0.2 * frames, np.zeros(n_frames)])
    drift = Drift(curve, method="rcc")
    table = Localizations({"x_nm": np.zeros(n_frames), "y_nm": np.zeros(n_frames),
                           "frame": frames}, {})
    return save_drift_corrected(tmp_path / name, table, drift), curve


def test_another_files_drift_is_applied_frame_by_frame(tmp_path):
    path, curve = _drift_file(tmp_path, 1000)
    frames = np.arange(0, 1000, 7)
    locs = Localizations({"x_nm": np.full(len(frames), 500.0),
                          "y_nm": np.full(len(frames), 800.0),
                          "frame": frames}, {})
    s = TransformSettings(source="drift", drift_path=str(path))
    out = plugins.get(PATH)().run(Context(locs=locs), s).locs
    np.testing.assert_allclose(out["x_nm"], 500.0 - curve[frames, 0], rtol=1e-6)
    np.testing.assert_allclose(out["y_nm"], 800.0 - curve[frames, 1], rtol=1e-6)
    assert out.metadata["transformations"][-1]["drift_of"] == str(path)


def test_another_files_drift_is_refused_for_other_frames(tmp_path):
    path, _ = _drift_file(tmp_path, 1000)
    plugin = plugins.get(PATH)()
    s = TransformSettings(source="drift", drift_path=str(path))
    for frames in (np.arange(0, 1500), np.arange(0, 500)):
        locs = Localizations({"x_nm": np.zeros(len(frames)),
                              "y_nm": np.zeros(len(frames)), "frame": frames}, {})
        with pytest.raises(ValueError, match="different acquisitions"):
            plugin.run(Context(locs=locs), s)
    check_same_frames(1000, np.arange(0, 995))       # a few frames short is fine


def test_a_drift_referenced_to_the_last_frame_is_zero_there():
    curve = np.column_stack([np.linspace(3, 40, 50), np.linspace(-5, 9, 50),
                             np.zeros(50)])
    drift = Drift(curve)
    last = drift.referenced("last frame")
    np.testing.assert_allclose(last.drift[-1], 0.0)
    np.testing.assert_allclose(np.diff(last.drift, axis=0), np.diff(curve, axis=0))
    np.testing.assert_allclose(drift.referenced("first frame").drift[0], 0.0)
    assert drift.referenced("as estimated") is drift
    assert Drift.from_dict(last.to_dict()).reference == "last frame"


def test_rcc_referenced_to_the_last_frame_is_zero_there():
    from smappy.rcc import RCCSettings, estimate_drift_rcc
    from smappy.simulate import simulate
    locs = simulate()
    drift = estimate_drift_rcc(locs, RCCSettings(n_timepoints=5,
                                                 reference="last frame"))
    np.testing.assert_allclose(drift.drift[-1], 0.0, atol=1e-9)
    assert drift.reference == "last frame"
    as_estimated = estimate_drift_rcc(locs, RCCSettings(n_timepoints=5))
    np.testing.assert_allclose(as_estimated.drift - as_estimated.drift[-1],
                               drift.drift, atol=1e-9)


# ------------------------------------------------------------------- the GUI
@pytest.fixture
def app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_the_main_window_opens_transform_on_a_named_transformation(app, tmp_path,
                                                                    monkeypatch):
    """Register layers' way in (step 5b): the "transform" window opener with
    `settings_for`; the form offers the session's names and image layers."""
    monkeypatch.setenv("SMAPPY_CONFIG_DIR", str(tmp_path))
    from smappy.gui.app import ControlWindow, RenderWindow
    session = Session(points())
    session.add_image(ImageData(np.ones((4, 4), np.float32), 10.0, name="wf"))
    session.add_transformation("cam2", typed((5, 0, 0)))
    window = ControlWindow(session, RenderWindow(session))
    session.window_openers["transform"](settings_for("cam2"))
    panel = window._plugin_windows[PATH].panel
    values = panel.form.value()
    assert values.source == "named" and values.named == "cam2"
    state = plugins.get(PATH)().describe(session.context(grouping="ungrouped"),
                                         values)
    assert (2, "image layer 2: wf") in state.choices["target"]
    assert state.choices["named"] == [("cam2", "cam2")]
    # the data's, not the GUI's: the workspace does not carry it
    assert "transformations" not in window.collect_workspace().to_dict()
    window.close()
