"""Export Stack: movies along frames and stacks along z, on 6a's renderer."""
import threading

import numpy as np
import pytest

from smappy.movie import (estimate_size, frame_steps, has_encoder, raw_plane, restrict,
                          z_steps)
from smappy.picture import read_description
from smappy.plugins import Context
from smappy.plugins.file import (ExportImageSettings, ExportStack, ExportStackSettings,
                                 StackFramesSettings, StackZSettings, export_image,
                                 stack_plan)
from smappy.session import Session

tifffile = pytest.importorskip("tifffile")
Image = pytest.importorskip("PIL.Image")


@pytest.fixture(scope="module")
def simulated():
    from smappy.simulate import simulate
    return simulate(n_frames=2000, seed=3)


def session_of(locs) -> Session:
    session = Session(locs)
    session.layers[0].show_grouped(False)
    return session


BARE = dict(scale_bar=False, colour_bars=False, labels=False, stamp=False)


def test_the_steps_of_a_movie_and_of_a_z_stack():
    steps = frame_steps(0, 999, 250)
    assert [(s.lo, s.hi) for s in steps] == [(0, 250), (250, 500), (500, 750), (750, 1000)]
    running = frame_steps(0, 999, 250, window=500)        # SMAP's running window
    assert [(s.lo, s.hi) for s in running] == [(0, 500), (250, 750), (500, 1000)]
    grown = frame_steps(0, 999, 250, cumulative=True)
    assert [(s.lo, s.hi) for s in grown] == [(0, 250), (0, 500), (0, 750), (0, 1000)]
    assert len(frame_steps(0, 10, 1000)) == 1             # one frame at least
    slices = z_steps(-300, 300, 100, thickness=200)
    assert [(s.lo, s.hi) for s in slices] == [(-300, -100), (-200, 0), (-100, 100),
                                              (0, 200), (100, 300)]


def test_each_movie_frame_is_the_picture_of_its_window(tmp_path, simulated):
    session = session_of(simulated)
    path = tmp_path / "m.tif"
    settings = ExportStackSettings(path=str(path), region="all", pixelsize_nm=40.0,
                                   layout="composite",
                                   frames=StackFramesSettings(step=500), **BARE)
    ExportStack().run(session.context(), settings)
    stack = tifffile.imread(path)
    last = int(np.max(simulated["frame"]))
    assert len(stack) == len(frame_steps(0, last, 500)) == 4
    for k, (lo, hi) in enumerate([(0, 499), (500, 999), (1000, 1499), (1500, 1999)]):
        session.layers[0].set_bound("frame", lo, hi)
        picture, = export_image(session.context(), ExportImageSettings(
            path=str(tmp_path / f"w{k}.png"), what="composite", region="all",
            pixelsize_nm=40.0, scale_bar=False, colour_bars=False, labels=False))
        with Image.open(picture) as image:
            assert np.array_equal(stack[k], np.asarray(image))
    with tifffile.TiffFile(path) as tif:
        assert tif.series[0].axes == "TYXS"
        assert tif.pages[0].resolution[0] == pytest.approx(1000 / 40.0)
    found = read_description(path)
    assert found["stack"]["steps"][1] == [500.0, 1000.0]
    assert found["export"]["frames"]["step"] == 500


def test_a_cumulative_movie_grows(simulated):
    session = session_of(simulated)
    settings = ExportStackSettings(path="x.tif", region="all", pixelsize_nm=40.0,
                                   frames=StackFramesSettings(step=400, cumulative=True))
    plan = stack_plan(session.context(), settings)
    shown = [int(restrict(plan.layers, "frame", s)[0].mask.sum()) for s in plan.steps]
    assert all(b > a for a, b in zip(shown, shown[1:]))
    assert shown[-1] == int(plan.layers[0].mask.sum())          # everything, at the end
    frames = list(plan.frames(plan.steps))
    lit = [int((f.max(axis=2) > 32).sum()) for f in frames]
    assert lit[-1] > lit[0]


def test_each_slice_holds_only_its_z_range(tmp_path, simulated):
    session = session_of(simulated)
    settings = ExportStackSettings(path=str(tmp_path / "z.tif"), along="z", region="all",
                                   pixelsize_nm=40.0, layout="composite",
                                   z=StackZSettings(step_nm=200.0, z_min_nm=-400.0,
                                                    z_max_nm=400.0), **BARE)
    plan = stack_plan(session.context(), settings)
    assert len(plan.steps) == 4
    z = np.asarray(plan.layers[0].locs["z_nm"])
    for step in plan.steps:
        shown, = restrict(plan.layers, "z_nm", step)
        inside = z[shown.mask]
        assert len(inside) and inside.min() >= step.lo and inside.max() < step.hi
    ExportStack().run(session.context(), settings)
    stack = tifffile.imread(tmp_path / "z.tif")
    session.layers[0].set_bound("z_nm", 0.0, 199.999)
    picture, = export_image(session.context(), ExportImageSettings(
        path=str(tmp_path / "s.png"), what="composite", region="all", pixelsize_nm=40.0,
        scale_bar=False, colour_bars=False, labels=False))
    with Image.open(picture) as image:
        assert np.array_equal(stack[2], np.asarray(image))
    with tifffile.TiffFile(tmp_path / "z.tif") as tif:
        assert tif.series[0].axes == "ZYXS"
        assert tif.imagej_metadata["spacing"] == pytest.approx(0.2)    # µm


def test_a_track_is_drawn_with_its_tail_up_to_now():
    from smappy.render import RenderSettings
    from smappy.simulate import simulate_tracks
    from smappy.locs import Localizations
    truth = simulate_tracks(n_frames=300, track_frames=80, density_um2=0.01, seed=2)
    locs = Localizations({**truth.columns, "track_id": truth["track_id_true"]},
                         truth.metadata)
    session = session_of(locs)
    session.layers[0].state.settings = RenderSettings(mode="tracks")
    settings = ExportStackSettings(path="t.tif", region="all", pixelsize_nm=50.0,
                                   frames=StackFramesSettings(step=10, tail=50))
    plan = stack_plan(session.context(), settings)
    frame = np.asarray(plan.layers[0].locs["frame"])
    step = plan.steps[10]                                  # frames 100 to 109
    with_tail, = restrict(plan.layers, "frame", step, tail=50)
    window, = restrict(plan.layers, "frame", step)
    assert frame[with_tail.mask].min() == 60 and frame[with_tail.mask].max() == 109
    assert frame[window.mask].min() == 100
    # the tracks are drawn longer with the tail than cut to the window
    image = list(plan.frames([step]))[0]
    plan.settings.frames.tail = None
    cut = list(plan.frames([step]))[0]
    assert (image.max(axis=2) > 32).sum() > 1.5 * (cut.max(axis=2) > 32).sum()


def test_the_camera_frames_go_beside_the_picture(simulated):
    from smappy.images import ImageData
    session = session_of(simulated)
    (x0, _), (y0, _) = session.full_view()
    planes = np.stack([np.full((60, 60), v, np.float32) for v in (9, 1, 2, 3)])
    planes += np.random.default_rng(0).uniform(0, 0.5, planes.shape).astype(np.float32)
    raw = ImageData(planes, 250.0, x0, y0, name="raw", kind="raw",
                    frames=np.array([-1, 0, 1000, 1900]))
    session.images.append(raw)
    assert raw_plane([raw], 1100)[1] == 2 and raw_plane([raw], 10)[1] == 1  # never the average
    settings = ExportStackSettings(path="r.tif", region="all", pixelsize_nm=40.0,
                                   layout="raw", frames=StackFramesSettings(step=500),
                                   **BARE)
    plan = stack_plan(session.context(), settings)
    image = list(plan.frames(plan.steps[:1]))[0]
    assert image.shape == (plan.fov.ny, 2 * plan.fov.nx + 2, 3) == plan.shape + (3,)
    assert image[:, :plan.fov.nx].any()                     # the camera, on the left
    session.images.clear()
    with pytest.raises(ValueError, match="no camera frames"):
        stack_plan(session.context(), settings)


@pytest.mark.parametrize("suffix", [".gif", ".mp4"])
def test_a_gif_and_an_mp4_are_written(tmp_path, simulated, suffix):
    if suffix == ".mp4" and not has_encoder():
        pytest.skip("imageio-ffmpeg is not installed")
    session = session_of(simulated)
    path = tmp_path / f"m{suffix}"
    settings = ExportStackSettings(path=str(path), region="all", pixelsize_nm=33.0,
                                   frames=StackFramesSettings(step=250, cumulative=True))
    result = ExportStack().run(session.context(), settings)
    n = result.data["frames"]
    assert n == 8
    plan = stack_plan(session.context(), settings)
    h, w = plan.shape
    if suffix == ".gif":
        with Image.open(path) as gif:
            assert gif.n_frames == n and gif.size == (w, h)
        assert read_description(path)["stack"]["along"] == "frames"
    else:
        import imageio_ffmpeg
        reader = imageio_ffmpeg.read_frames(str(path))
        meta = reader.__next__()
        assert meta["size"] == (w + w % 2, h + h % 2)          # even sides
        assert sum(1 for _ in reader) == n
        # the size the form shows is the right order of magnitude
        size = estimate_size("mp4", h, w, n, settings.quality)
        assert size / 4 < path.stat().st_size < size * 4


def test_the_form_says_how_many_frames_and_how_large(simulated):
    session = session_of(simulated)
    settings = ExportStackSettings(path="m.tif", region="all", pixelsize_nm=40.0,
                                   frames=StackFramesSettings(step=500))
    state = ExportStack().describe(session.context(), settings)
    plan = stack_plan(session.context(), settings)
    h, w = plan.shape
    assert state.notice.text.startswith(f"4 frames of {w} x {h}")
    assert state.shown["frames"] and not state.shown["z"] and not state.shown["quality"]
    settings.path, settings.along = "m.mp4", "z"
    state = ExportStack().describe(session.context(), settings)
    assert state.shown["z"] and state.shown["quality"]
    assert ExportStack().preflight(session.context(), ExportStackSettings(
        path="m.tif", region="all", pixelsize_nm=40.0)) is None   # quick: no question


def test_a_stopped_stack_leaves_no_file(tmp_path, simulated):
    from smappy.plugins import Stopped
    session = session_of(simulated)
    ctx = session.context()
    ctx.stop = threading.Event()
    ctx.stop.set()
    path = tmp_path / "s.tif"
    with pytest.raises(Stopped):
        ExportStack().run(ctx, ExportStackSettings(path=str(path), region="all",
                                                   pixelsize_nm=40.0))
    assert not path.exists()
