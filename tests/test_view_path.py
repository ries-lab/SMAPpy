"""The player's view path: key views, interpolated about the point looked at."""
import math

import numpy as np
import pytest

from smappy import view_path as vp
from smappy.view3d import Projection, euler_zxz


def viewpoint(az=0.0, el=0.0, centre=(0.0, 0.0, 0.0), zoom=10.0):
    return vp.Viewpoint.of(Projection(azimuth=az, elevation=el, pivot=np.array(centre, float),
                                   zoom=zoom))


def test_quaternions_go_there_and_back_for_any_turn():
    rng = np.random.default_rng(1)
    for _ in range(50):
        p = Projection(*rng.uniform(0, 360, 3))
        np.testing.assert_allclose(vp.matrix(vp.quaternion(p.matrix)), p.matrix, atol=1e-12)


def test_a_key_is_the_view_about_the_middle_of_the_screen():
    p = Projection(azimuth=30, elevation=50, pivot=np.array([100.0, 200.0, 0.0]),
                   offset=np.array([300.0, -50.0]), zoom=5.0)
    view = vp.Viewpoint.of(p)
    np.testing.assert_allclose(view.centre, p.screen_centre())
    q = view.apply(Projection(opacity=0.7))
    assert q.opacity == 0.7 and np.allclose(q.offset, 0)
    # orthographic, the picture is the same: every point lands where it did
    pts = np.random.default_rng(0).uniform(-1000, 1000, (3, 20))
    a, b = p.apply(*pts), q.apply(*pts)
    np.testing.assert_allclose(np.array(a[:2]) - p.offset[:, None], b[:2], atol=1e-9)


def test_a_whole_turn_about_z_turns_at_a_constant_rate_and_closes():
    start = viewpoint(el=60.0)
    path = vp.ViewPath([vp.Key(start)] + vp.turn_keys(start, "z", 360.0, seconds=12.0),
                         ease=False)
    assert len(path.keys) == 5 and path.seconds == pytest.approx(12.0) and path.closed
    for t in np.linspace(0, 1, 13):
        R = path.at(t).rotation
        # the data's z keeps its tilt; the azimuth goes down by 360 t
        az, el, _ = euler_zxz(R)
        assert el == pytest.approx(60.0, abs=1e-6)
        assert (-az) % 360 == pytest.approx((360 * t) % 360, abs=1e-6) or t in (0, 1)
    # a loop leaves out the last frame, which is the first again
    assert path.samples(4, loop=True).tolist() == [0, 0.25, 0.5, 0.75]
    assert path.samples(3).tolist() == [0, 0.5, 1]


def test_the_zoom_goes_evenly_on_a_log_scale_and_the_centre_on_a_line():
    a = viewpoint(centre=(0, 0, 0), zoom=16.0)
    b = vp.Viewpoint(a.rotation, np.array([1000.0, 0.0, 0.0]), 4.0)
    c = vp.Viewpoint(a.rotation, np.array([2000.0, 0.0, 0.0]), 1.0)
    path = vp.ViewPath([vp.Key(a), vp.Key(b, 1.0), vp.Key(c, 1.0)], ease=False)
    for t in np.linspace(0, 1, 9):
        view = path.at(t)
        assert view.zoom == pytest.approx(16.0 * 16.0 ** -t)
        np.testing.assert_allclose(view.centre, [2000.0 * t, 0, 0], atol=1e-9)


def test_ease_starts_and_ends_slowly_but_passes_the_middle():
    start = viewpoint()
    path = vp.ViewPath([vp.Key(start)] + vp.turn_keys(start, "screen vertical", 90.0,
                                                        seconds=1.0))
    early = vp.ViewPath(path.keys, ease=False).at(0.05).rotation
    eased = path.at(0.05).rotation

    def angle(R):
        return math.degrees(math.acos(np.clip((np.trace(R @ start.rotation.T) - 1) / 2, -1, 1)))

    assert angle(eased) < angle(early) / 3
    assert angle(path.at(0.5).rotation) == pytest.approx(45.0, abs=1e-6)


def test_a_turn_with_a_zoom_ends_turned_and_zoomed():
    start = viewpoint(zoom=8.0)
    keys = vp.turn_keys(start, "x", 180.0, zoom_by=4.0, seconds=6.0)
    assert len(keys) == 2 and keys[-1].view.zoom == pytest.approx(2.0)
    np.testing.assert_allclose(keys[-1].view.rotation @ np.array([0, 1.0, 0]),
                               [0, -1.0, 0], atol=1e-12)
    zoom_only = vp.turn_keys(start, "z", 0.0, zoom_by=2.0, seconds=3.0)
    assert len(zoom_only) == 1 and zoom_only[0].view.zoom == pytest.approx(4.0)


def test_a_path_survives_being_written_out(tmp_path):
    start = viewpoint(az=20, el=40, zoom=3.0)
    path = vp.ViewPath([vp.Key(start)] + vp.turn_keys(start, "y", 120.0), ease=False)
    again = vp.ViewPath.from_dict(path.to_dict())
    assert again.ease is False and len(again.keys) == len(path.keys)
    np.testing.assert_allclose(again.at(0.3).rotation, path.at(0.3).rotation, atol=1e-9)
    vp.save(path, tmp_path / "turn.view.yaml")
    loaded = vp.load(tmp_path / "turn.view.yaml")
    np.testing.assert_allclose(loaded.at(0.7).centre, path.at(0.7).centre)


def test_the_turn_through_keys_set_by_hand_does_not_jerk_at_a_key():
    """A turn about the screen's vertical, then about its horizontal: slerp
    from key to key changes its angular velocity at the middle key at once;
    the path's curve carries it through."""
    start = viewpoint(el=30.0)
    mid = vp.Viewpoint(vp.turned(start.rotation, "screen vertical", 60.0), start.centre, 10.0)
    end = vp.Viewpoint(vp.turned(mid.rotation, "screen horizontal", 60.0), start.centre, 10.0)
    path = vp.ViewPath([vp.Key(start), vp.Key(mid, 2.0), vp.Key(end, 2.0)], ease=False)
    assert np.allclose(path.at(0.5).rotation, mid.rotation)          # passes the key

    def velocity(t, dt=1e-4):
        # the axis-angle of the turn over a small step, per unit of t
        a, b = vp.quaternion(path.at(t).rotation), vp.quaternion(path.at(t + dt).rotation)
        return 2 * vp.qlog(vp.qmul(b, vp.qconj(a))) / dt

    np.testing.assert_allclose(velocity(0.5 - 2e-4), velocity(0.5 + 1e-4), atol=0.02)
    # where slerp alone would turn about one axis and then about another
    w0, w1 = velocity(0.5 - 2e-4), velocity(0.5 + 1e-4)
    assert np.linalg.norm(w0) > 1.0 and np.linalg.norm(w1) > 1.0


def test_a_key_keeps_the_box_and_the_front_face_moves_through_the_sample():
    from smappy.view3d import Slab
    start = vp.Viewpoint.of(Projection(), Slab([0, 0, 0], [2000, 2000, 800]))
    # the next key's box 200 deep, its back face where it was
    end = vp.Viewpoint(start.rotation, start.centre, start.zoom, None,
                       Slab([0, 0, -300], [2000, 2000, 200]))
    path = vp.ViewPath([vp.Key(start), vp.Key(end, 4.0)], ease=False)
    assert path.moves_box and not vp.ViewPath([vp.Key(start), vp.Key(start, 1.0)]).moves_box
    for t in np.linspace(0, 1, 9):
        box = path.at(t).slab
        assert box.center[2] - box.size[2] / 2 == pytest.approx(-400.0)      # the back face
        assert box.center[2] + box.size[2] / 2 == pytest.approx(400.0 - 600.0 * t)
    again = vp.ViewPath.from_dict(path.to_dict())
    assert again.at(0.5).slab.size[2] == pytest.approx(500.0)
