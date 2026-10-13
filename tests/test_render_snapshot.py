"""Renders run on a snapshot of the layers, never on the layers themselves.

The render threads used to be handed the view widget (2D) or the session
(3D) and walk the live layers from their own thread, so whatever the GUI
thread changed meanwhile -- a filter, a setting, the table, a block from a
live fit -- was read half-way.  A snapshot is taken on the GUI thread
(`LayerSnapshot`); these check that it is frozen, that what reaches a worker
is a snapshot and nothing with a Qt object or the session in it, and that a
live fit can go on appending while one is drawn.
"""
import dataclasses
import threading

import numpy as np
import pytest

from smappy.images import ImageData
from smappy.layer_state import LayerSnapshot, render_panels, snapshot_layers
from smappy.locs import Localizations
from smappy.render import FieldOfView
from smappy.session import Layer, Session
from smappy.view3d import Projection, render_3d, snapshot_3d


def table(n=20_000, seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    return Localizations({
        "x_nm": (rng.uniform(0, 8000, n) + shift).astype(np.float32),
        "y_nm": rng.uniform(0, 8000, n).astype(np.float32),
        "z_nm": rng.normal(0, 150, n).astype(np.float32),
        "frame": rng.integers(0, 5000, n).astype(np.int64),
        "photons": rng.uniform(100, 3000, n).astype(np.float32),
        "xy_err_nm": rng.uniform(4, 20, n).astype(np.float32),
    }, {"units": "nm"})


def block(frame, n=400):
    rng = np.random.default_rng(frame)
    return Localizations({"x_nm": rng.uniform(0, 8000, n).astype(np.float32),
                          "y_nm": rng.uniform(0, 8000, n).astype(np.float32),
                          "frame": np.full(n, frame, np.int64),
                          "photons": rng.uniform(100, 3000, n).astype(np.float32),
                          "xy_err_nm": rng.uniform(4, 20, n).astype(np.float32)}, {})


FOV = FieldOfView.fit((0, 8000), (0, 8000), 160, 160)
ZOOMED = FieldOfView.fit((3000, 4000), (3000, 4000), 160, 160)    # the index culls


def two_layer_session():
    session = Session(table())
    session.layers[0].show_grouped(False)
    session.add_layer(like=0)
    session.layers[1].set_bound("photons", 1500, None)
    session.add_image(ImageData(np.ones((8, 8), np.float32), 1000.0, name="wf"))
    return session


def test_a_snapshot_renders_the_layers_as_they_were_when_it_was_taken():
    session = two_layer_session()
    snap = snapshot_layers(session.layers)
    panels = (0, 1, 2, "overlay")
    before = [render_panels(snap, fov, panels) for fov in (FOV, ZOOMED)]

    # everything the GUI thread can change while a render runs, the settings
    # edited in place as the panels edit them
    first = session.layers[0]
    first.set_bound("photons", 2500, None)
    first.state.settings.mode = "hist"
    first.state.display.contrast = 1.0
    first.state.display.lut = "gray"
    session.layers[1].visible = False
    session.layers[2].image.x0 = 4000.0
    session.layers[2].frame = 3
    session.locs = table(seed=1, shift=3000.0)

    again = [render_panels(snap, fov, panels) for fov in (FOV, ZOOMED)]
    for (images, weight), (images2, weight2) in zip(before, again):
        for a, b in zip(images, images2):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(weight, weight2)
    # ...and the session itself has moved on: a fresh snapshot draws that
    now, _ = render_panels(snapshot_layers(session.layers), FOV, panels)
    assert not np.allclose(now[0], before[0][0][0])


def test_the_3d_view_renders_a_snapshot_as_it_was_taken():
    session = two_layer_session()
    slab = session.slab_from_roi(use_roi=False)
    projection = Projection(azimuth=30, elevation=20)
    projection.fit(slab, 200, 160)
    fov = projection.fov(200, 160)
    snap = snapshot_3d(session.layers)
    rgb, hist = render_3d(snap, projection, slab, fov)

    session.layers[0].set_bound("photons", 2500, None)
    session.layers[0].state.settings.mode = "hist"
    session.locs = table(seed=1, shift=3000.0)

    rgb2, hist2 = render_3d(snap, projection, slab, fov)
    np.testing.assert_array_equal(rgb, rgb2)
    np.testing.assert_array_equal(hist, hist2)
    # session layers are snapshotted on the way in, for a script
    rgb3, _ = render_3d(session.layers, projection, slab, fov)
    assert not np.allclose(rgb, rgb3)


def test_snapshots_of_one_table_share_its_identity_for_the_3d_caches():
    """Every drag frame is a new snapshot; the GPU buffers and the row cache
    are keyed on the table's token, so the snapshots must carry the same one."""
    from smappy.view3d import _table_key
    session = two_layer_session()
    a, b = snapshot_3d(session.layers), snapshot_3d(session.layers)
    assert a[0].locs is not b[0].locs
    assert _table_key(a[0].locs) == _table_key(b[0].locs)


def test_a_snapshot_is_references_not_copies():
    """Taking one costs nothing at any size: the columns, the mask and the
    index are the layer's own arrays; only the settings are copied."""
    session = two_layer_session()
    layer = session.layers[1]
    snap = layer.state.snapshot()
    for name, column in layer.locs.columns.items():
        assert snap.locs[name] is column
    assert snap.mask is layer.filter.mask
    assert snap.index is layer.state.index
    assert snap.settings == layer.state.settings and snap.settings is not layer.state.settings


def test_indices_compacted_on_the_render_thread_are_kept_by_the_filter():
    session = two_layer_session()
    layer = session.layers[1]
    layer.set_bound("photons", 1000, None)
    assert layer.filter.cached_indices is None
    snap = layer.state.snapshot()
    rows = snap.indices                       # the render thread's work
    layer.state.keep_indices(snap)
    assert layer.filter.cached_indices is rows
    # a filter changed meanwhile does not take indices of the old mask
    layer.set_bound("photons", 2000, None)
    old = layer.state.snapshot()
    layer.set_bound("photons", 2500, None)
    old.indices
    layer.state.keep_indices(old)
    assert layer.filter.cached_indices is None


def live_session(blocks=3):
    session = Session()
    session.begin_live((0, 8000, 0, 8000))
    for frame in range(1, blocks + 1):
        session.append(block(frame))
    return session


def test_a_snapshot_of_a_live_table_keeps_its_rows_while_blocks_arrive():
    session = live_session()
    snap = snapshot_layers(session.layers)
    n = len(snap[0].locs)
    before = [render_panels(snap, fov)[1] for fov in (FOV, ZOOMED)]
    index = session.layers[0].state.index
    for frame in range(4, 40):              # through reallocations and merges
        session.append(block(frame))
    assert index.n_segments < 36            # the live index has merged meanwhile
    assert len(snap[0].locs) == len(snap[0].mask) == n
    assert snap[0].index.n_localizations == n
    assert snap[0].select(FOV).max() < n
    after = [render_panels(snap, fov)[1] for fov in (FOV, ZOOMED)]
    for a, b in zip(before, after):
        np.testing.assert_array_equal(a, b)


def test_a_live_fit_appends_while_a_snapshot_is_being_rendered():
    """The render thread draws the snapshot over and over while this thread
    appends -- as the GUI thread does while a render is running."""
    session = live_session()
    snap = snapshot_layers(session.layers)
    expected = [render_panels(snap, fov)[1] for fov in (FOV, ZOOMED)]
    errors, mismatches = [], []
    stop = threading.Event()

    def draw():
        try:
            while not stop.is_set():
                for fov, want in zip((FOV, ZOOMED), expected):
                    if not np.array_equal(render_panels(snap, fov)[1], want):
                        mismatches.append(fov)
        except Exception as e:          # pragma: no cover - the failure itself
            errors.append(e)

    worker = threading.Thread(target=draw)
    worker.start()
    try:
        for frame in range(4, 120):
            session.append(block(frame))
    finally:
        stop.set()
        worker.join(60)
    assert not errors and not mismatches
    assert len(session.locs) == 119 * 400


# ------------------------------------------------------------------ the GUI
pytest.importorskip("PySide6")


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def settle(app, until=lambda: False, rounds=300):
    import time
    for _ in range(rounds):
        app.processEvents()
        if until():
            return
        time.sleep(0.01)


def reachable(obj, seen=None):
    """Everything an object holds, through containers, dataclasses and
    attributes -- what a worker could reach from what it was handed."""
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return
    seen.add(id(obj))
    yield obj
    if isinstance(obj, (np.ndarray, str, bytes, int, float, bool, type(None))):
        return
    if isinstance(obj, dict):
        items = list(obj.keys()) + list(obj.values())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        items = list(obj)
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        items = [getattr(obj, f.name) for f in dataclasses.fields(obj)]
        items += list(getattr(obj, "__dict__", {}).values())
    else:
        items = list(getattr(obj, "__dict__", {}).values())
    for item in items:
        yield from reachable(item, seen)


def assert_no_qt_and_no_session(args):
    from PySide6.QtCore import QObject
    from smappy.filter import LocFilter
    from smappy.layer_state import LocSet, Tables, ViewState
    for obj in reachable(args):
        assert not isinstance(obj, QObject), f"a worker was handed {obj!r}"
        assert not isinstance(obj, (Session, Layer, ViewState, Tables, LocSet, LocFilter)), \
            f"a worker was handed {type(obj).__name__}"
        module = type(obj).__module__ or ""
        assert not module.startswith(("PySide6", "shiboken", "pyqtgraph")), \
            f"a worker was handed {type(obj).__name__} from {module}"


def spy_on_worker(monkeypatch):
    """What `RenderWorker.render` -- the slot, on the worker's thread -- receives,
    and which worker received it: every view has its own, and a view an
    earlier test left open (collection is off) may still be rendering."""
    from smappy.gui.render_view import RenderWorker
    seen = []
    real = RenderWorker.render

    def render(self, job, generation):
        seen.append((self, job, generation))
        return real(self, job, generation)

    monkeypatch.setattr(RenderWorker, "render", render)
    return seen


def test_the_2d_worker_is_handed_a_snapshot_and_no_qt_object(app, monkeypatch):
    from smappy.gui.render_view import PanelJob, RenderView
    seen = spy_on_worker(monkeypatch)
    session = two_layer_session()
    view = RenderView(session)
    view.resize(300, 300)
    view.show()
    view.set_layout_mode("side+overlay")
    view.tile_valid = False
    view.render()
    mine = lambda: [(j, g) for w, j, g in seen if w is view._renderer]
    settle(app, lambda: mine() and not view._busy)
    assert mine() and view.fov is not None and not view.last_error
    for job, generation in mine():
        assert isinstance(job, PanelJob) and isinstance(generation, int)
        assert all(isinstance(layer, LayerSnapshot) for layer in job.layers)
    assert_no_qt_and_no_session(mine())
    view.close()


def test_the_3d_worker_is_handed_a_snapshot_and_no_qt_object(app, monkeypatch):
    from smappy.gui.view3d import Job3D, View3D
    seen = spy_on_worker(monkeypatch)
    session = two_layer_session()
    view = View3D(session)
    view.resize(300, 300)
    view.show()
    view.fit()
    view.render(preview=True)
    mine = lambda: [(j, g) for w, j, g in seen if w is view._renderer]
    settle(app, lambda: mine() and not view._busy and view._fov is not None)
    assert mine() and view._fov is not None
    for job, generation in mine():
        assert isinstance(job, Job3D) and isinstance(generation, int)
        assert all(isinstance(layer, LayerSnapshot) for layer in job.layers)
    assert_no_qt_and_no_session(mine())
    view.close()


def test_a_failed_3d_render_frees_the_view_for_the_next(app, monkeypatch):
    """It used to print and return without answering, and the view waited
    for that render for good."""
    from smappy.gui import view3d
    from smappy.gui.view3d import View3D
    session = two_layer_session()
    view = View3D(session)
    view.resize(300, 300)
    view.show()
    # showing it renders already, and queues more (the timer, a render asked
    # for while one ran): let all of them finish before breaking one, or a
    # queued render that succeeds runs after the broken one
    def idle():
        return not view._busy and view._pending is None and not view._timer.isActive()
    settle(app, idle)
    assert idle()
    view._fov = None
    real = view3d.render_3d
    calls = {"n": 0}

    def once_broken(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("cannot draw")
        return real(*args, **kwargs)

    monkeypatch.setattr(view3d, "render_3d", once_broken)
    view.render()
    settle(app, lambda: calls["n"] >= 1 and idle())
    assert calls["n"] == 1 and idle() and view._fov is None
    view.render()
    settle(app, lambda: not view._busy and view._fov is not None)
    assert view._fov is not None
    view.close()


def test_the_view_keeps_rendering_while_a_live_fit_appends(app):
    from smappy.gui.render_view import RenderView
    session = live_session()
    view = RenderView(session)
    view.resize(300, 300)
    view.show()
    for frame in range(4, 60):
        session.append(block(frame))            # the view schedules a render
        view.tile_valid = False
        view.render()                           # and one is often still running
        app.processEvents()
    settle(app, lambda: not view._busy and not view._timer.isActive())
    view.tile_valid = False
    view.render()
    settle(app, lambda: not view._busy)
    assert not view.last_error and view.fov is not None
    assert float(np.max(view.image.image)) > 0
    view.close()
