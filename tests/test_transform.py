"""Transform: typed operations, files, images and another file's drift.

The typed operations are checked against coordinates worked out by hand, the
rest against what they are for: a _2ct.h5 brings the second camera onto the
first, an image resampled by a transformation is bright where the localizations
moved by it land -- the secondary half of a split chip on the main one, by a
bead map or the move a two-channel fit records -- another file's drift corrects
a file of the same frames and refuses one of other frames.
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
from smappy.session import Session
from smappy.transformation import (Transformation, check_same_frames,
                                   load_transformation, save_transformation,
                                   typed)

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
def _blobs(plane, rows, cols):
    """3 x 3 squares of ones: bilinear sampling of one is 1 near its middle."""
    for r, c in zip(rows, cols):
        plane[r - 1:r + 2, c - 1:c + 2] = 1.0


def test_an_image_moved_lands_where_the_localizations_land():
    """Bright squares at the localizations, moved by one transformation as an
    image and as localizations: the new image is bright where the moved
    localizations are, and the original stays as it was."""
    rng = np.random.default_rng(1)
    pixel = 20.0
    plane = np.zeros((60, 80), np.float32)
    rows, cols = rng.integers(8, 52, 20), rng.integers(8, 72, 20)
    _blobs(plane, rows, cols)
    x = (cols + 0.5) * pixel + 300.0      # the pixel centres, image at x0=300
    y = (rows + 0.5) * pixel + 100.0
    locs = Localizations({"x_nm": x, "y_nm": y, "frame": np.zeros(len(x), int)}, {})
    session = Session(locs)
    session.add_image(ImageData(plane.copy(), pixel, x0=300.0, y0=100.0, name="wf"))
    s = TransformSettings(typed=TypedSettings(angle_deg=10, scale=1.05,
                                              mirror="x", shift_x_nm=-70,
                                              centre="given", centre_x_nm=1100,
                                              centre_y_nm=700))
    plugin = plugins.get(PATH)()
    original = session.layers[1].image
    session.run(plugin, dataclasses.replace(s, target=2))    # the image layer
    session.run(plugin, s)                                   # the localizations
    np.testing.assert_array_equal(original.data, plane)      # left as it was
    moved = session.layers[1].image                          # the layer shows the move
    assert moved.name == "wf_T"
    assert len(session.layers) == 2 and any(im is original for im in session.images)
    values = moved.values_at(np.asarray(session.locs["x_nm"]),
                             np.asarray(session.locs["y_nm"]))
    inside = ((np.asarray(session.locs["x_nm"]) > 320) & (np.asarray(session.locs["x_nm"]) < 1880)
              & (np.asarray(session.locs["y_nm"]) > 120) & (np.asarray(session.locs["y_nm"]) < 1280))
    assert inside.sum() > 10
    assert values[inside].min() > 0.9
    assert moved.data.sum() < 0.2 * moved.data.size        # and not everywhere


def _split_chip(tmp_path, perspective=2e-6):
    """A 64 x 128 chip, main channel left, secondary right, and a projective
    map of the secondary onto the main kept in a _2ct.h5 with its regions;
    squares in the main half and their partners where the map sends them
    from in the secondary."""
    from smappy.calibrate.dual import ChannelTransform, map_points
    from smappy.calibrate.transform import save_channel_transform
    angle = np.radians(1.0)
    h = np.array([[np.cos(angle), np.sin(angle), -64.5],
                  [-np.sin(angle), np.cos(angle), 1.5],
                  [perspective, 0.0, 1.0]])
    regions = [[0, 0, 64, 64], [64, 0, 64, 64]]
    path = tmp_path / "beads_2ct.h5"
    save_channel_transform(path, ChannelTransform(h, {"channels": {"regions": regions}}, {}))
    rng = np.random.default_rng(3)
    main = np.column_stack([rng.integers(8, 56, 15), rng.integers(8, 56, 15)])  # (col, row)
    partner = np.rint(map_points(np.linalg.inv(h), main.astype(float))).astype(int)
    main = np.rint(map_points(h, partner.astype(float))).astype(int)  # on whole pixels both
    plane = np.zeros((64, 128), np.float32)
    _blobs(plane, main[:, 1], main[:, 0])
    _blobs(plane, partner[:, 1], partner[:, 0])
    pixel = 100.0
    # pixel i centred on i * pixel, as a localization's x_nm is
    image = ImageData(plane, pixel, x0=-0.5 * pixel, y0=-0.5 * pixel, name="wf")
    return path, h, regions, image, main * pixel


def test_a_projective_bead_map_moves_only_the_secondary_half_onto_the_main(tmp_path):
    path, h, regions, image, main_nm = _split_chip(tmp_path)
    session = Session(points())
    session.add_image(image)
    other = session.add_image(ImageData(np.ones((4, 4), np.float32), 100.0, name="bf"))
    session.add_image(image)                         # a second layer of the same image
    session.run(plugins.get(PATH)(), TransformSettings(target=2, source="file",
                                                       path=str(path)))
    moved = session.layers[1].image
    assert session.layers[3].image is moved           # every layer that showed it
    assert other.image.name == "bf"                   # and only those
    kept = next(im for im in session.images if im.name == "wf_main")
    assert moved.name == "wf_T" and len(session.layers) == 4
    x, y = main_nm[:, 0], main_nm[:, 1]
    assert moved.values_at(x, y).min() > 0.8           # the partners on the main ones
    assert kept.values_at(x, y).min() == 1.0
    assert not moved.data[:, 64:].any() and not kept.data[:, 64:].any()


def test_the_move_a_two_channel_fit_records_moves_its_image(tmp_path):
    """Overlay channels records its map with the table; Transform's *recorded*
    source finds it on auto and moves the image of the same chip by it."""
    from smappy.calibrate.transform import load_transform
    from smappy.channel_overlay import overlay_table
    path, h, regions, image, main_nm = _split_chip(tmp_path)
    fitted = Localizations({"x_nm": np.array([100.0, 7000.0]), "y_nm": np.array([100.0, 200.0]),
                            "frame": np.zeros(2, int), "camera_channel": np.array([0, 1])},
                           {"pixelsize_nm": 100.0})
    session = Session(overlay_table(fitted, load_transform(path), path=path))
    session.add_image(image)
    plugin = plugins.get(PATH)()
    s = TransformSettings(target=2, source="recorded")
    state = plugin.describe(Context(session=session), s)
    assert any("overlay channels" in label for _, label in state.choices["recorded"])
    session.run(plugin, s)
    moved = session.layers[1].image
    assert moved.values_at(main_nm[:, 0], main_nm[:, 1]).min() > 0.8


def test_an_image_without_channel_regions_says_to_move_it_whole(tmp_path):
    session = Session(points())
    session.add_image(ImageData(np.ones((10, 10), np.float32), 10.0, name="wf"))
    p = Transformation("projective", np.array([[1, 0, 5], [0, 1, 0], [1e-6, 0, 1]]))
    session.add_transformation("bent", p)
    plugin = plugins.get(PATH)()
    with pytest.raises(ValueError, match="whole image"):
        session.run(plugin, settings_for("bent", target=2))
    session.run(plugin, dataclasses.replace(settings_for("bent", target=2), part="all"))
    assert session.layers[1].image.name == "wf_T"
    with pytest.raises(ValueError, match="no z"):
        session.run(plugin, TransformSettings(target=2, typed=TypedSettings(shift_z_nm=5)))


def test_the_moved_image_is_saved_as_a_tiff_with_its_pixel_size(tmp_path):
    from smappy.images import load_images
    session = Session(points())
    session.add_image(ImageData(np.arange(12, dtype=np.float32).reshape(3, 4), 10.0, name="wf"))
    session.run(plugins.get(PATH)(), TransformSettings(
        target=2, typed=TypedSettings(shift_x_nm=10, centre="origin"),
        image_to=str(tmp_path / "wf_T.tif")))
    back = load_images(tmp_path / "wf_T.tif")[0]
    assert back.pixelsize == pytest.approx(10.0)
    np.testing.assert_allclose(back.data, session.layers[1].image.data)


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


def test_an_image_an_old_file_placed_is_resampled_where_it_was_placed(tmp_path):
    """Transform once moved an image by an affine placement kept in the file's
    attributes; such a file opens with the image resampled to where it was."""
    import h5py
    from smappy.io.hdf5 import load_images
    plane = np.zeros((20, 30), np.float32)
    _blobs(plane, [5], [8])                           # around pixel (row 5, col 8)
    session = Session(points())
    session.add_image(ImageData(plane, 10.0, x0=0.0, y0=0.0, name="wf"))
    path = session.save(tmp_path / "old.hdf5")
    turn = typed(angle_deg=90.0, centre=(0.0, 0.0))   # x, y -> y, -x
    with h5py.File(path, "a") as f:
        m = turn.matrix
        f["images/0"].attrs["placement"] = np.array(
            [[m[0, 0], m[0, 1], m[0, 3]], [m[1, 0], m[1, 1], m[1, 3]], [0, 0, 1.0]])
    back = load_images(path)[0]
    x, y, _ = turn.apply(np.array([85.0]), np.array([55.0]))   # the blob's centre, placed
    assert back.values_at(x, y)[0] == pytest.approx(1.0)
    assert back.bounds == pytest.approx((0.0, -300.0, 200.0, 0.0))
