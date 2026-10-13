"""The GUI opened on a running fit, as a program driving the microscope opens it.

The dataset grows here the way NDTiffStorage writes one -- pixels appended to
the stack file, then a record appended to the index -- so the fit is reading
while the "microscope" writes, and the pause in the middle is longer than the
fit's idle timeout: only the writer's word may end it.
"""
import json
import struct
import time

import numpy as np
import pytest

from smappy.io.ndtiff import INDEX_NAME, SUMMARY_MARKER, SUMMARY_OFFSET

PLUGIN = "Localize/Gaussian 2D"


class GrowingNDTiff:
    """An NDTiff dataset written a frame at a time, by appending."""

    NAME = "Stack_NDTiffStack.tif"

    def __init__(self, folder, shape):
        self.folder = folder
        folder.mkdir(parents=True)
        summary = json.dumps({"PixelSize_um": 0.1, "Height": shape[0],
                              "Width": shape[1]}).encode()
        (folder / self.NAME).write_bytes(
            b"\0" * SUMMARY_OFFSET + struct.pack("<II", SUMMARY_MARKER, len(summary))
            + summary)
        (folder / INDEX_NAME).write_bytes(b"")
        self.n = 0

    def write(self, frames):
        for frame in frames:
            stack = self.folder / self.NAME
            offset = stack.stat().st_size
            pixels = np.ascontiguousarray(frame, dtype=np.uint16)
            metadata = json.dumps({"Core-Camera": "Camera", "Camera-Offset": "100",
                                   "ROI": f"0-0-{frame.shape[1]}-{frame.shape[0]}",
                                   "Exposure-ms": 20.0}).encode()
            with stack.open("ab") as f:
                f.write(pixels.tobytes() + metadata)
            axes = json.dumps({"time": self.n}).encode()
            record = (struct.pack("<I", len(axes)) + axes
                      + struct.pack("<I", len(self.NAME)) + self.NAME.encode()
                      + struct.pack("<8I", offset, frame.shape[1], frame.shape[0], 1,
                                    0, offset + pixels.nbytes, len(metadata), 0))
            with (self.folder / INDEX_NAME).open("ab") as f:
                f.write(record)          # last: the image is complete before it
            self.n += 1


@pytest.fixture(scope="module")
def frames():
    from smappy.simulate import LabellingSettings, camera_frames
    stack, _ = camera_frames(n_frames=40, size_px=32, seed=3,
                             labelling=LabellingSettings(efficiency=0.1))
    return stack


def settings_for(folder, out, **more):
    from smappy.gui.live_session import fit_settings
    values = {"source.path": str(folder), "source.live": True,
              "source.live_timeout": 0.5, "source.chunk": 10,
              "camera.conversion": 0.5, "output.path": str(out),
              "output.raw_frames": 0}
    values.update(more)
    return fit_settings(PLUGIN, values)


def pump(until, seconds=60.0):
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    deadline = time.monotonic() + seconds
    while not until() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.02)
    return until()


def close(live):
    for window in (live.control, live.render):
        window.close()


def test_a_misspelt_setting_is_refused_rather_than_left_at_its_default():
    from smappy.gui.live_session import fit_settings
    with pytest.raises(ValueError, match="camera.ofset"):
        fit_settings(PLUGIN, {"camera.ofset": 100})
    assert fit_settings(PLUGIN, {"camera.offset": 99.0}).camera.offset == 99.0


def test_without_a_window_the_fit_ends_on_the_writers_word(tmp_path, frames):
    """The headless path: the plugin run from a script with a `Context`."""
    import threading

    from smappy import plugins
    from smappy.plugins import Context

    data = GrowingNDTiff(tmp_path / "ds", frames.shape[1:])
    data.write(frames)
    finished = threading.Event()
    finished.set()                       # a saved dataset: complete from the start
    out = tmp_path / "locs.hdf5"
    result = plugins.get(PLUGIN)().run(Context(writer_finished=finished),
                                       settings_for(data.folder, out,
                                                    **{"source.live_timeout": 30.0}))
    assert result.data["stats"]["frames"] == len(frames)
    assert not result.stopped and out.exists()


@pytest.mark.slow
def test_the_fit_waits_out_a_pause_and_ends_when_the_writer_has_finished(tmp_path, frames):
    pytest.importorskip("PySide6")
    from smappy.gui.live_session import LiveSession
    from smappy.io.hdf5 import load_localizations

    data = GrowingNDTiff(tmp_path / "ds", frames.shape[1:])
    data.write(frames[:10])
    out = tmp_path / "out" / "locs.hdf5"
    out.parent.mkdir()
    ended = []
    live = LiveSession(PLUGIN, settings_for(data.folder, out), on_finished=ended.append)
    live.open()
    try:
        # four idle timeouts with nothing new: a hand-started live fit
        # would have stopped here
        assert not pump(lambda: ended, seconds=2.0)
        data.write(frames[10:])
        live.writer_finished.set()
        assert pump(lambda: ended)
        outcome, = ended
        assert (outcome.state, outcome.complete) == ("succeeded", True)
        assert outcome.stats["frames"] == len(frames)
        assert outcome.path == out
        saved = load_localizations(out)
        assert len(saved) == len(live.session.locs) > 0

        # the file is handed on as it is; the session's work goes elsewhere
        assert live.session.is_protected(out)
        with pytest.raises(ValueError, match="another name"):
            live.session.save(out)
        live.session.save(tmp_path / "out" / "edited.hdf5")
    finally:
        close(live)
        live.wait()


@pytest.mark.slow
def test_stopping_keeps_what_was_fitted_and_says_the_input_is_incomplete(tmp_path, frames):
    pytest.importorskip("PySide6")
    from smappy.gui.live_session import LiveSession
    from smappy.io.hdf5 import load_localizations

    data = GrowingNDTiff(tmp_path / "ds", frames.shape[1:])
    data.write(frames[:20])
    out = tmp_path / "locs.hdf5"
    ended = []
    live = LiveSession(PLUGIN, settings_for(data.folder, out), on_finished=ended.append)
    live.open()
    try:
        assert pump(lambda: len(live.session.locs) > 0)
        live.stop.set()
        assert pump(lambda: ended)
        outcome, = ended
        assert (outcome.state, outcome.complete) == ("cancelled", False)
        assert len(load_localizations(out)) == len(live.session.locs)
    finally:
        close(live)
        live.wait()


@pytest.mark.slow
def test_closing_the_windows_while_it_fits_is_a_cancellation(tmp_path, frames):
    pytest.importorskip("PySide6")
    from smappy.gui.live_session import LiveSession

    data = GrowingNDTiff(tmp_path / "ds", frames.shape[1:])
    data.write(frames[:10])
    ended = []
    live = LiveSession(PLUGIN, settings_for(data.folder, tmp_path / "locs.hdf5"),
                       on_finished=ended.append)
    live.open()
    assert pump(lambda: len(live.session.locs) > 0)      # fitting, not starting
    close(live)
    live.wait()                          # what `run` does once the loop ends
    outcome, = ended
    assert outcome.state == "cancelled" and not outcome.complete
    assert (tmp_path / "locs.hdf5").exists()


@pytest.mark.slow
def test_a_fit_that_cannot_start_is_reported_once_as_failed(tmp_path):
    pytest.importorskip("PySide6")
    from smappy.gui.live_session import LiveSession

    ended = []
    settings = settings_for(tmp_path / "nothing-here", tmp_path / "locs.hdf5",
                            **{"source.live": False})
    live = LiveSession(PLUGIN, settings, on_finished=ended.append)
    live.open()
    try:
        assert pump(lambda: ended)
        outcome, = ended
        assert outcome.state == "failed" and outcome.error
    finally:
        close(live)
        live.wait()
    assert len(ended) == 1


def _fit_in_background(folder, out, **more):
    """A headless live fit on a thread; its streamed blocks and warnings, timed."""
    import threading

    from smappy import plugins
    from smappy.plugins import Context

    seen = {"blocks": [], "warnings": []}

    def stream(event, payload):
        if event == "block":
            seen["blocks"].append((time.monotonic(), np.unique(payload["frame"])))
        elif event == "warning":
            seen["warnings"].append(payload)

    finished = threading.Event()
    context = Context(stream=stream, writer_finished=finished)
    thread = threading.Thread(target=plugins.get(PLUGIN)().run,
                              args=(context, settings_for(folder, out, **more)))
    thread.start()
    return seen, finished, thread


def test_a_frame_is_fitted_within_a_fraction_of_a_second_of_being_written(tmp_path, frames):
    """Each read is fitted at once: no waiting for boxes to gather or a 5 s timer."""
    data = GrowingNDTiff(tmp_path / "ds", frames.shape[1:])
    data.write(frames[:1])
    seen, finished, thread = _fit_in_background(data.folder, tmp_path / "locs.hdf5")
    try:
        deadline = time.monotonic() + 20
        while not seen["blocks"] and time.monotonic() < deadline:
            time.sleep(0.01)              # the first block: the fit is running
        lags = []
        for frame in frames[1:]:
            data.write([frame])
            written = time.monotonic()
            n = data.n - 1
            while time.monotonic() < written + 3:
                hit = [t for t, which in seen["blocks"] if n in which]
                if hit:
                    lags.append(hit[0] - written)
                    break
                time.sleep(0.005)
    finally:
        finished.set()
        thread.join(30)
    assert len(lags) > 20
    assert np.median(lags) < 0.5, lags


def test_a_fit_that_falls_behind_the_camera_says_so(tmp_path, frames, monkeypatch):
    import smappy.plugins.fit as fit

    monkeypatch.setattr(fit, "LIVE_BEHIND_WARN_SECONDS", 0.0)
    data = GrowingNDTiff(tmp_path / "ds", frames.shape[1:])
    data.write(frames)                  # all of it before the fit reads a frame
    seen, finished, thread = _fit_in_background(data.folder, tmp_path / "locs.hdf5",
                                                **{"source.chunk": 2})
    finished.set()
    thread.join(30)
    assert len(seen["warnings"]) == 1
    assert "cannot keep up" in seen["warnings"][0]


def test_the_window_hands_the_fits_progress_on_as_numbers(tmp_path, frames):
    """`on_stream` gets the fitted blocks and a ``fitted`` count per block:
    what a program following the fit (smappy-microclaw) turns into metrics."""
    pytest.importorskip("PySide6")
    from smappy.gui.live_session import LiveSession

    data = GrowingNDTiff(tmp_path / "ds", frames.shape[1:])
    data.write(frames)
    streamed = []
    ended = []
    live = LiveSession(PLUGIN, settings_for(data.folder, tmp_path / "locs.hdf5"),
                       on_finished=ended.append,
                       on_stream=lambda event, payload: streamed.append((event, payload)))
    live.open()
    try:
        live.writer_finished.set()
        assert pump(lambda: ended)
    finally:
        close(live)
        live.wait()
    fitted = [p for e, p in streamed if e == "fitted"]
    blocks = [p for e, p in streamed if e == "block"]
    assert fitted and blocks
    assert fitted[-1]["frames"] == len(frames)
    assert fitted[-1]["localizations"] == sum(len(b) for b in blocks) == len(live.session.locs)
    assert [f["frames"] for f in fitted] == sorted(f["frames"] for f in fitted)
    assert fitted[-1]["behind"] == 0        # caught up: nothing written is unread
