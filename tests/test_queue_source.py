"""Frames pushed in from elsewhere, read as if they were being written to disk.

The point of `QueueSource` is that the live machinery cannot tell the
difference, so the tests are mostly about the block boundaries: what a reader is
handed must be the frames it says it is, and a slow producer must not have its
last few frames held back waiting for a full block.
"""
import threading
import time

import numpy as np
import pytest

from smappy.io.queue_source import QueueSource, queue_source
from smappy.io.watch import WatchSettings

FAST = WatchSettings(poll=0.01, timeout=0.5)
SHAPE = (6, 5)


def frame(value: int) -> np.ndarray:
    return np.full(SHAPE, value, np.uint16)


def read(source, **kw) -> list:
    return [(start, block) for start, block in
            source.watch(settings=FAST, **kw)]


def test_pushed_frames_come_back_in_order():
    source = queue_source(shape=SHAPE)
    for i in range(5):
        source.push(frame(i))
    source.close()

    blocks = read(source, chunk=2)
    assert [(start, len(b)) for start, b in blocks] == [(0, 2), (2, 2), (4, 1)]
    assert np.array_equal(np.concatenate([b for _, b in blocks])[:, 0, 0],
                          np.arange(5))


def test_a_block_may_be_pushed_at_once():
    source = queue_source(shape=SHAPE)
    source.push(np.stack([frame(i) for i in range(4)]))
    source.close()
    (start, block), = read(source, chunk=8)
    assert start == 0 and len(block) == 4


def test_the_producer_may_number_the_frames_itself():
    """An acquisition hook knows the real frame index; it must survive."""
    source = queue_source(shape=SHAPE)
    source.push(frame(0), first_frame=100)
    source.push(frame(1))
    source.close()

    (start, block), = read(source, chunk=8)
    assert start == 100 and len(block) == 2
    assert source.n_frames == 102


def test_a_gap_in_the_numbering_ends_the_block():
    """A block claims to be contiguous frames, so it has to be."""
    source = queue_source(shape=SHAPE)
    source.push(np.stack([frame(0), frame(1)]))
    source.push(frame(2), first_frame=50)
    source.close()

    assert [(start, len(b)) for start, b in read(source, chunk=8)] == [(0, 2), (50, 1)]


def test_a_slow_producer_is_not_held_back():
    """Waiting for a full block would sit on the last frames indefinitely."""
    source = queue_source(shape=SHAPE)

    def produce():
        for i in range(3):
            time.sleep(0.05)
            source.push(frame(i))
        source.close()

    thread = threading.Thread(target=produce)
    thread.start()
    blocks = read(source, chunk=100)      # far more than will ever arrive
    thread.join()

    assert sum(len(b) for _, b in blocks) == 3
    assert len(blocks) > 1                # handed over as they came, not at the end


def test_closing_ends_the_stream_and_pushing_after_it_is_an_error():
    source = queue_source(shape=SHAPE)
    source.push(frame(0))
    source.close()
    assert sum(len(b) for _, b in read(source)) == 1
    with pytest.raises(ValueError, match="push after close"):
        source.push(frame(1))


def test_a_stop_event_ends_it_at_once():
    source = queue_source(shape=SHAPE)
    stop = threading.Event()
    stop.set()
    assert read(source, stop_event=stop) == []


def test_start_and_stop_bound_the_frames():
    source = queue_source(shape=SHAPE)
    source.push(np.stack([frame(i) for i in range(10)]))
    source.close()
    blocks = read(source, chunk=100, start=3, stop=7)
    (start, block), = blocks
    assert start == 3
    assert np.array_equal(block[:, 0, 0], np.arange(3, 7))


def test_the_first_frame_settles_the_shape_if_it_was_not_given():
    source = queue_source()
    assert source.shape is None
    source.push(frame(0))
    assert source.shape == SHAPE and source.dtype == np.uint16


def test_it_holds_no_past():
    source = queue_source(shape=SHAPE)
    source.push(frame(0))
    with pytest.raises(NotImplementedError, match="handed on as they arrive"):
        source.frame(0)


def test_a_bounded_queue_makes_the_producer_wait():
    """The only honest answer when the fit cannot keep up."""
    import queue as _queue

    source = queue_source(shape=SHAPE, maxsize=1)
    source.push(frame(0))
    with pytest.raises(_queue.Full):
        source.push(frame(1), timeout=0.05)


def test_the_live_fit_reads_it_like_a_growing_file():
    """The whole point: `LiveFit` cannot tell where the frames came from."""
    from smappy.detect import AbsoluteCutoff, DoGFilter, PeakFinder
    from smappy.live import LiveFit, LiveSettings
    from smappy.metadata import CameraMetadata
    from smappy.pipeline import FitSettings
    from smappy.psf import GaussianPSF

    shape = (48, 48)
    rng = np.random.default_rng(0)
    source = queue_source(shape=shape)

    def emitter(i):
        y, x = np.mgrid[0:shape[0], 0:shape[1]]
        cx, cy = rng.uniform(8, 40), rng.uniform(8, 40)
        clean = 2000.0 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / 2.0)
        return (rng.poisson(clean) + 100).astype(np.uint16)

    def acquire():
        for i in range(6):
            time.sleep(0.02)
            source.push(emitter(i))
        source.close()

    live = LiveSettings(chunk=2, flush_seconds=0.05,
                        watch=WatchSettings(poll=0.01, timeout=0.5))
    fit = LiveFit(source.watch(chunk=live.chunk, settings=live.watch),
                  CameraMetadata(conversion=1.0, offset=100.0, pixelsize_um=0.1),
                  PeakFinder(DoGFilter(1.2), AbsoluteCutoff(40.0)),
                  GaussianPSF(sigma=1.2),
                  FitSettings(roisize=9, output_unit="nm"), live)
    thread = threading.Thread(target=acquire)
    thread.start()
    fit.start()
    fit.finished.wait(10)
    thread.join()

    assert fit.error is None
    assert fit.n_emitted == 6
    assert fit.engine.stats["frames"] == 6


def test_live_view_takes_a_queue_source(tmp_path):
    """The live window, fed by a producer instead of by a file."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from smappy.detect import AbsoluteCutoff, DoGFilter, PeakFinder
    from smappy.io.hdf5 import load_localizations
    from smappy.live import LiveSettings, live_view
    from smappy.metadata import CameraMetadata
    from smappy.pipeline import FitSettings
    from smappy.psf import GaussianPSF

    shape = (48, 48)
    rng = np.random.default_rng(1)
    source = queue_source(shape=shape)
    out = tmp_path / "live.h5"

    def acquire():
        for _ in range(6):
            time.sleep(0.02)
            y, x = np.mgrid[0:shape[0], 0:shape[1]]
            cx, cy = rng.uniform(8, 40), rng.uniform(8, 40)
            clean = 2000.0 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / 2.0)
            source.push((rng.poisson(clean) + 100).astype(np.uint16))
        source.close()

    live = LiveSettings(chunk=2, update_seconds=0.05, flush_seconds=0.05,
                        watch=WatchSettings(poll=0.01, timeout=0.5))
    thread = threading.Thread(target=acquire)
    thread.start()
    viewer = live_view(source, CameraMetadata(conversion=1.0, offset=100.0,
                                              pixelsize_um=0.1),
                       PeakFinder(DoGFilter(1.2), AbsoluteCutoff(40.0)),
                       GaussianPSF(sigma=1.2),
                       FitSettings(roisize=9, output_unit="nm"),
                       output=out, live=live, block=False)
    try:
        deadline = time.monotonic() + 20
        while viewer.fit.running and time.monotonic() < deadline:
            viewer.update()
            time.sleep(0.02)
        viewer.update()
        thread.join()

        assert viewer.fit.error is None
        assert len(viewer.state.locs) == 6
        # the frame comes from the camera, so the image does not rescale
        assert viewer.axes.get_xlim()[1] > shape[1] * 100.0 * 0.9
    finally:
        viewer.close()
        viewer.fit.stop()
        plt.close(viewer.figure)

    assert len(load_localizations(out)) == 6
